# src/model_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import logging
import random
import asyncio 

from src.model_parts import CombineContext, DenoiseNet, ProjectHead, UpdateFastState
from src.text_decoder import TextDecoder
from src.utils import initialize_weights, decode_sequence
from src.emotion import EmotionalCore
from src.heart import Heart
from src.self_reflection import SelfReflectionModule
from src.self_prompting import SelfPromptingModule

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ContinuouslyReasoningPredictor(nn.Module):
    """
    The Continuously Reasoning Predictor (CRP) model.
    This model integrates a reasoning diffusion process with continuous learning,
    governed by a rhythmic EmotionalCore and regulated by a homeostatic Heart.
    """
    def __init__(self, vocab_size: int, sos_token_id: int, eos_token_id: int, pad_token_id: int, device: torch.device, input_dims: int = None):
        super().__init__()
        self.device = device
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.emotions = EmotionalCore().to(device)
        self.heart = Heart().to(device)

        self.model_dims = self.emotions.model_dims.int().item()
        self.knowledge_dims = self.emotions.knowledge_dims.int().item()

        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_model.to(device)
        self.embedding_dim = input_dims if input_dims is not None else self.embedding_model.get_sentence_embedding_dimension()

        if self.embedding_dim != self.model_dims:
            self.input_projection = nn.Linear(self.embedding_dim, self.model_dims).to(device)
            initialize_weights(self.input_projection)
        else:
            self.input_projection = nn.Identity().to(device)

        self.slow_state = nn.Parameter(torch.randn(1, self.knowledge_dims, device=device) * 0.01)
        self.fast_state = nn.Parameter(torch.zeros(1, self.model_dims, device=device), requires_grad=False)

        self.combine_context = CombineContext(
            current_input_dims=self.model_dims,
            fast_state_dims=self.model_dims,
            slow_state_dims=self.knowledge_dims,
            context_dims=self.model_dims
        ).to(device)

        self.denoise_net = DenoiseNet(
            model_dims=self.model_dims,
            context_dims=self.model_dims,
            time_embedding_dim=self.emotions.time_embedding_dim.int().item()
        ).to(device)

        self.project_head = ProjectHead(
            model_dims=self.model_dims,
            output_dims=self.embedding_dim
        ).to(device)

        self.update_fast_state_func = UpdateFastState(
            fast_state_dims=self.model_dims,
            current_input_dims=self.model_dims,
            reasoned_state_z0_dims=self.model_dims
        ).to(device)

        self.text_decoder = TextDecoder(
            embedding_dim=self.embedding_dim,
            hidden_size=self.model_dims,
            vocab_size=vocab_size,
            sos_token_id=sos_token_id
        ).to(device)

        self.self_reflection_module = SelfReflectionModule(
            model_dims=self.model_dims,
            reflection_dims=self.model_dims
        ).to(device)

        self.self_prompting_module = SelfPromptingModule().to(device)

        # Variables to store the latest calculated metrics
        self.latest_confidence = None
        self.latest_meta_error = None
        self.latest_state_drift = None
        self.latest_heart_metrics = {}


        logger.info(f"ContinuouslyReasoningPredictor initialized with Resonant Heart architecture.")


    async def _reason_to_predict(self, model_dims_input_embedding: torch.Tensor, stop_event: asyncio.Event = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = model_dims_input_embedding.shape[0]
        current_slow_state = self.slow_state.expand(batch_size, -1)
        current_fast_state = self.fast_state.expand(batch_size, -1)

        c_t = self.combine_context(model_dims_input_embedding, current_fast_state, current_slow_state)
        z_tau = torch.randn(batch_size, self.model_dims, device=self.device)

        T_DIFF_STEPS = self.emotions.get_focus()

        for tau_int in reversed(range(1, T_DIFF_STEPS + 1)):
            if stop_event and stop_event.is_set():
                raise asyncio.CancelledError("Training stopped by user.")
            tau_tensor = torch.tensor([tau_int], dtype=torch.float32, device=self.device)
            z_tau = self.denoise_net(z_tau, tau_tensor, c_t)
            await asyncio.sleep(0)
        z_0 = z_tau
        return z_0, c_t


    async def _get_predicted_embedding(self, input_embedding: torch.Tensor, stop_event: asyncio.Event = None) -> torch.Tensor:
        self.emotions.update()
        
        processed_input_embedding = self.input_projection(input_embedding)

        reasoned_state_z0, c_t_returned = await self._reason_to_predict(processed_input_embedding, stop_event)

        batch_size = processed_input_embedding.shape[0]
        expanded_fast_state_for_reflection = self.fast_state.expand(batch_size, -1)
        current_confidence, current_meta_error = self.self_reflection_module(
            reasoned_state=reasoned_state_z0,
            fast_state=expanded_fast_state_for_reflection,
            slow_state=self.slow_state
        )

        self.latest_confidence = current_confidence
        self.latest_meta_error = current_meta_error
        
        with torch.no_grad():
            similarity = F.cosine_similarity(self.slow_state, self.fast_state.mean(dim=0, keepdim=True))
            self.latest_state_drift = (1.0 - similarity.mean()).item()

        # Capture the dictionary of metrics returned by the Heart
        self.latest_heart_metrics = self.heart.beat(self.emotions, self.latest_confidence, self.latest_meta_error)

        predicted_embedding = self.project_head(reasoned_state_z0)

        with torch.no_grad():
            expanded_fast_state = self.fast_state.expand(batch_size, -1)
            new_fast_state = self.update_fast_state_func(
                expanded_fast_state,
                processed_input_embedding,
                reasoned_state_z0
            ).mean(dim=0, keepdim=True)
            self.fast_state.copy_(new_fast_state)

        return predicted_embedding


    async def learn_one_step(self, x_t: torch.Tensor, target_sequence_t_plus_1: torch.Tensor, stop_event: asyncio.Event = None):
        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Training stopped by user.")

        predicted_embedding = await self._get_predicted_embedding(x_t, stop_event)

        max_len = target_sequence_t_plus_1.shape[1]
        decoded_logits = self.text_decoder.forward_teacher_forced(predicted_embedding, max_len, target_sequence_t_plus_1)

        loss_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss_value = loss_criterion(
            decoded_logits.reshape(-1, self.text_decoder.vocab_size),
            target_sequence_t_plus_1.reshape(-1)
        )
        return loss_value

    async def generate_text(self, input_embedding: torch.Tensor, max_len: int = None, top_p: float = 0.9) -> list[int]:
        if max_len is None:
            max_len = self.emotions.max_seq_len.int().item()

        self.eval()
        with torch.no_grad():
            predicted_embedding = await self._get_predicted_embedding(input_embedding)
            
            batch_size = predicted_embedding.shape[0]
            hidden = self.text_decoder.get_initial_hidden(predicted_embedding)
            
            input_token = torch.full((batch_size,), self.sos_token_id, dtype=torch.long, device=self.device)
            generated_sequences = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            
            for t in range(max_len):
                logits, hidden = self.text_decoder.forward(input_token, hidden)
                
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
                next_token_probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1).squeeze(1)

                generated_sequences[:, t] = next_token
                input_token = next_token

                if (next_token == self.eos_token_id).all():
                    break
        
        return generated_sequences.tolist()


    async def generate_internal_thought(self, vocab, max_len: int = 64, input_prompt_override: str = None) -> tuple[str, float, float, int, float, str]:
        self.eval()
        with torch.no_grad():
            if input_prompt_override is not None:
                self_reflection_prompt = input_prompt_override
            else:
                current_states = {
                    "confidence": self.latest_confidence.mean().item() if self.latest_confidence is not None else 0.0,
                    "meta_error": self.latest_meta_error.mean().item() if self.latest_meta_error is not None else 0.0,
                    "focus": self.emotions.get_focus(),
                    "curiosity": self.emotions.get_curiosity()
                }
                self_reflection_prompt = self.self_prompting_module.generate_prompt(current_states)

            prompt_embedding = self.embedding_model.encode([self_reflection_prompt], convert_to_tensor=True, device=self.device)
            
            generated_ids_list = await self.generate_text(prompt_embedding, max_len=max_len)
            thought_text = decode_sequence(generated_ids_list[0], vocab, self.eos_token_id)

            confidence = self.latest_confidence.mean().item() if self.latest_confidence is not None else 0.0
            meta_error = self.latest_meta_error.mean().item() if self.latest_meta_error is not None else 0.0
            focus = self.emotions.get_focus()
            curiosity = self.emotions.get_curiosity()

            logger.info(f"Internal Prompt: '{self_reflection_prompt}' -> Thought: '{thought_text}' - Conf: {confidence:.4f}, ME: {meta_error:.4f}, F: {focus}, C: {curiosity:.6f}")

            return thought_text, confidence, meta_error, focus, curiosity, self_reflection_prompt