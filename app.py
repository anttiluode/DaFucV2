import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import json
import logging
import gradio as gr
import time
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VAE Class for Latent Space Encoding (unchanged)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)
        z = self.sample_latent(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

# Modified FractalNode class without Superweights
class FractalNode(nn.Module):
    def __init__(self, input_dim, output_dim, depth=0, max_depth=5, max_children=2):
        super().__init__()
        self.traditional_weight = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.traditional_weight.weight)
        self.norm = nn.LayerNorm(output_dim)
        self._children = []
        self.is_active = True
        self.max_children = max_children
        self.complexity_threshold = 0.5
        self.depth = depth
        self.max_depth = max_depth
        self.attention_weights = nn.Parameter(torch.ones(max_children))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        base_output = self.traditional_weight(x)
        base_output = self.norm(base_output)
        complexity = self.calculate_complexity(base_output)

        if complexity > self.complexity_threshold and len(self._children) < self.max_children and self.depth < self.max_depth:
            new_child = FractalNode(self.traditional_weight.out_features, self.traditional_weight.out_features, 
                                    depth=self.depth+1, max_depth=self.max_depth)
            self._children.append(new_child)
            self.add_module(f'child_{len(self._children)}', new_child)

        modulated_output = base_output

        for i, child in enumerate(self._children):
            if child.is_active:
                child_output = child(modulated_output)
                modulated_output = modulated_output + child_output * F.softmax(self.attention_weights, dim=0)[i]

        return modulated_output

    def calculate_complexity(self, output):
        return torch.log(1 + torch.norm(output))

    def calculate_relevance(self, child_output):
        return torch.sigmoid(torch.sum(child_output))

    def grow(self, complexity_threshold):
        if self.calculate_complexity(self.traditional_weight.weight) > complexity_threshold and len(self._children) < self.max_children and self.depth < self.max_depth:
            new_child = FractalNode(self.traditional_weight.out_features, self.traditional_weight.out_features, 
                                    depth=self.depth+1, max_depth=self.max_depth)
            self._children.append(new_child)
            self.add_module(f'child_{len(self._children)}', new_child)
        for child in self._children:
            child.grow(complexity_threshold)

    def update_attention(self, co_activation_vector):
        self.attention_weights.data += co_activation_vector[:len(self._children)]
        self.attention_weights.data = F.softmax(self.attention_weights, dim=0)

    @property
    def complexity(self):
        return torch.norm(self.traditional_weight.weight)

    @property
    def children(self):
        return self._children

# Modified FUCWM class without Superweights
class FUCWM(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim, max_depth=5):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.root = FractalNode(embed_dim, output_dim, max_depth=max_depth)
        self.max_depth = max_depth
        self.co_activation_matrix = torch.zeros((max_depth, max_depth))

    def forward(self, x):
        if x.dtype == torch.long:
            embedded = self.word_embeddings(x)
            if embedded.dim() == 3:
                embedded = embedded.mean(dim=1)
        else:
            embedded = x
        
        output = self.root(embedded)
        self.update_co_activations()
        return output

    def grow(self, complexity_threshold):
        self.root.grow(complexity_threshold)

    def manage_padding(self):
        def _manage_padding(node, depth):
            if depth >= self.max_depth:
                node.is_active = False
            else:
                activation = torch.norm(node.traditional_weight.weight)
                if not node.is_active and activation > 0.5:
                    node.is_active = True
                elif node.is_active and activation < 0.1:
                    node.is_active = False
            for child in node.children:
                _manage_padding(child, depth + 1)
        _manage_padding(self.root, 0)

    def update_co_activations(self):
        for i in range(self.max_depth):
            for j in range(self.max_depth):
                if i != j:
                    self.co_activation_matrix[i][j] += 0.1 * random.random()
        
        self.co_activation_matrix = F.softmax(self.co_activation_matrix, dim=1)

    def update_attention_weights(self):
        def update_node(node, depth):
            node.update_attention(self.co_activation_matrix[depth])
            for child in node.children:
                update_node(child, depth+1)
        
        update_node(self.root, 0)

# DynamicAI class (mostly unchanged, removed superweight-related methods)
class DynamicAI:
    def __init__(self, vocab_size=10000, embed_dim=64, latent_dim=64, output_dim=64, max_depth=5):
        self.vae = VAE(embed_dim, latent_dim)
        self.model = FUCWM(vocab_size, embed_dim, output_dim, max_depth)
        self.optimizer = optim.Adam(list(self.vae.parameters()) + list(self.model.parameters()), lr=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.criterion = nn.MSELoss()
        self.word_to_index = {}
        self.index_to_word = {}
        self.next_index = 0
        self.lm_studio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def tokenize(self, text):
        words = text.lower().split()
        indices = []
        for word in words:
            if word not in self.word_to_index:
                self.word_to_index[word] = self.next_index
                self.index_to_word[self.next_index] = word
                self.next_index += 1
            indices.append(self.word_to_index[word])
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def save_state(self, filename):
        state = {
            'model_state': self.model.state_dict(),
            'vae_state': self.vae.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'next_index': self.next_index
        }
        torch.save(state, filename)
        logger.info(f"Model state saved to {filename}")

    def load_state(self, filename):
        state = torch.load(filename)
        self.word_to_index = state['word_to_index']
        self.index_to_word = state['index_to_word']
        self.next_index = state['next_index']
        
        self.rebuild_model_structure(state['model_state'])
        
        self.model.load_state_dict(state['model_state'])
        self.vae.load_state_dict(state['vae_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        
        logger.info(f"Model state loaded from {filename}")

    def rebuild_model_structure(self, state_dict):
        def rebuild_node(node, prefix):
            child_indices = set()
            for name in state_dict.keys():
                if name.startswith(prefix):
                    parts = name[len(prefix):].split('.')
                    if parts[0].startswith('child_'):
                        child_index = int(parts[0].split('_')[1])
                        child_indices.add(child_index)
            
            for index in sorted(child_indices):
                while len(node._children) < index:
                    new_child = FractalNode(node.traditional_weight.out_features, 
                                            node.traditional_weight.out_features, 
                                            depth=node.depth+1, 
                                            max_depth=node.max_depth)
                    node._children.append(new_child)
                    node.add_module(f'child_{len(node._children)}', new_child)
                
                child_prefix = f"{prefix}child_{index}."
                rebuild_node(node._children[index-1], child_prefix)

        rebuild_node(self.model.root, "root.")

    def chat(self, input_text, max_length=20, temperature=0.7):
        input_tokens = self.tokenize(input_text)
        thinking_process = []
        with torch.no_grad():
            embedded_q = self.model.word_embeddings(input_tokens)
            _, _, _, z_q = self.vae(embedded_q.mean(dim=1))
            output, node_info = self.fractal_thinking(z_q)
            thinking_process.append(node_info)

        response = []
        for _ in range(max_length):
            output = output / temperature
            probs = torch.softmax(output, dim=-1)
            next_word_index = torch.multinomial(probs, 1).item()
            next_word = self.index_to_word.get(next_word_index, "")
            if next_word:
                response.append(next_word)
                if next_word in ['.', '!', '?']:
                    break
                next_token = self.tokenize(next_word)
                _, _, _, next_latent = self.vae(self.model.word_embeddings(next_token).mean(dim=1))
                output, node_info = self.fractal_thinking(next_latent)
                thinking_process.append(node_info)
            else:
                break

        thinking_str = "\n".join(thinking_process)
        response_str = ' '.join(response)
        return f"Thinking Process:\n{thinking_str}\n\nResponse: {response_str}"

    def fractal_thinking(self, input_vector):
        def traverse_node(node, x, depth):
            node_info = f"Node depth: {depth}, Complexity: {node.complexity.item():.4f}, Children: {len(node.children)}"
            output = node(x)
            
            if depth < node.max_depth:
                for child in node.children:
                    child_output, child_info = traverse_node(child, output, depth + 1)
                    output = output + child_output * node.calculate_relevance(child_output)
                    node_info += f"\n{child_info}"

            return output, node_info

        output, node_info = traverse_node(self.model.root, input_vector, 0)
        return output, node_info

    def talk_with_lm_studio(self, initial_message, conversation_duration=60, delay=2):
        message = initial_message
        start_time = time.time()
        conversation_log = []

        while time.time() - start_time < conversation_duration:
            ai_response = self.chat(message)
            logger.info(f"DynamicAI:\n{ai_response}")
            conversation_log.append(f"DynamicAI:\n{ai_response}")
            yield "\n\n".join(conversation_log)

            ai_message = ai_response.split("Response: ")[-1].strip()

            if not ai_message:
                logger.info("DynamicAI generated an empty response. Skipping LM Studio turn.")
                conversation_log.append("DynamicAI: [No response generated. Still learning...]")
                yield "\n\n".join(conversation_log)
                time.sleep(delay)
                continue

            lm_studio_response = self.send_to_lm_studio(ai_message)
            if lm_studio_response:
                logger.info(f"LM Studio: {lm_studio_response}")
                conversation_log.append(f"LM Studio: {lm_studio_response}")
                message = lm_studio_response
                yield "\n\n".join(conversation_log)
            else:
                logger.warning("No response from LM Studio. Ending conversation.")
                break
            
            time.sleep(delay)

    def send_to_lm_studio(self, message):
        if not message.strip():
            logger.warning("Attempted to send an empty message to LM Studio. Skipping.")
            return None

        try:
            completion = self.lm_studio_client.chat.completions.create(
                model="unsloth/Llama-3.2-3B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "You're talking to an experimental AI that is still learning to communicate. If it doesn't respond or sends empty messages, please be patient and continue the conversation."},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            logger.error(f"Error sending to LM Studio: {str(e)}")
            return None


    def train_on_qa_pairs(self, qa_pairs, epochs=10):
        if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
            raise ValueError("qa_pairs must be a non-empty list")
        
        logger.info(f"Training on {len(qa_pairs)} Q&A pairs for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            errors = 0
            random.shuffle(qa_pairs)
            for i, (question, answer) in enumerate(qa_pairs):
                self.optimizer.zero_grad()
                
                try:
                    q_tokens = self.tokenize(question)
                    a_tokens = self.tokenize(answer)
                    
                    q_embedded = self.model.word_embeddings(q_tokens)
                    _, _, _, q_latent = self.vae(q_embedded.mean(dim=1))
                    
                    a_embedded = self.model.word_embeddings(a_tokens)
                    _, _, _, a_latent = self.vae(a_embedded.mean(dim=1))
                    
                    q_output = self.model(q_latent)
                    a_output = self.model(a_latent)
                    
                    loss = self.criterion(q_output, a_output)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    
                    self.optimizer.step()
                    
                    total_loss += loss.item()

                    self.model.grow(complexity_threshold=0.5)
                    self.model.manage_padding()
                    self.model.update_attention_weights()

                    if i % 10 == 0:
                        logger.info(f"Epoch {epoch+1}, Pair {i+1}/{len(qa_pairs)}, Loss: {loss.item():.4f}")

                except Exception as e:
                    logger.error(f"Error processing pair: {question} | {answer}")
                    logger.error(f"Error details: {str(e)}")
                    errors += 1
                    continue

            avg_loss = total_loss / (len(qa_pairs) - errors) if len(qa_pairs) > errors else 0
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Errors: {errors}")
            
            self.scheduler.step()
            
            self.save_state(f"model_state_epoch_{epoch+1}.pth")
            
            yield epoch + 1, avg_loss, errors

    def grow(self, complexity_threshold):
        self.model.grow(complexity_threshold)

    def manage_padding(self):
        self.model.manage_padding()

# Gradio Interface for DynamicAI
def create_gradio_interface(ai):
    def handle_chat(message, temperature):
        return ai.chat(message, temperature=float(temperature))

    def handle_save(filename):
        ai.save_state(filename)
        return f"State saved to {filename}"

    def handle_load(filename):
        ai.load_state(filename)
        return f"State loaded from {filename}"

    def handle_train_qa(qa_pairs_file, epochs):
        try:
            with open(qa_pairs_file.name, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            output = ["Starting training..."]
            for epoch, loss, errors in ai.train_on_qa_pairs(qa_pairs, epochs=int(epochs)):
                output.append(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Errors: {errors}")
            output.append("Training completed successfully")
            return "\n".join(output)
        except Exception as e:
            return f"Error during training: {str(e)}"

    def handle_lm_studio_chat(initial_message, duration, delay):
        conversation_log = gr.Textbox()
        for log in ai.talk_with_lm_studio(initial_message, conversation_duration=float(duration), delay=float(delay)):
            conversation_log = log
            yield conversation_log

    with gr.Blocks() as interface:
        gr.Markdown("# Dynamic AI with Fractal Universe Chocolate Wafer Model and Attention Mechanism")

        with gr.Tab("Chat"):
            chat_input = gr.Textbox(label="Your message")
            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
            chat_output = gr.Textbox(label="AI response")
            chat_button = gr.Button("Chat")
            chat_button.click(handle_chat, inputs=[chat_input, temperature], outputs=chat_output)

        with gr.Tab("LM Studio Conversation"):
            initial_message = gr.Textbox(label="Initial Message")
            duration = gr.Number(label="Conversation Duration (seconds)", value=60)
            delay = gr.Number(label="Delay between messages (seconds)", value=2)
            conversation_log = gr.Textbox(label="Conversation Log", lines=20)
            start_conversation = gr.Button("Start Conversation")
            start_conversation.click(handle_lm_studio_chat, inputs=[initial_message, duration, delay], outputs=conversation_log)

        with gr.Tab("Train on Q&A"):
            qa_file = gr.File(label="Q&A Pairs JSON File")
            epochs_input = gr.Number(label="Number of Epochs", value=10)
            train_button = gr.Button("Train on Q&A Pairs")
            train_output = gr.Textbox(label="Training status")
            train_button.click(handle_train_qa, inputs=[qa_file, epochs_input], outputs=train_output)

        with gr.Tab("Save/Load State"):
            filename_input = gr.Textbox(label="Filename")
            save_button = gr.Button("Save State")
            load_button = gr.Button("Load State")
            state_output = gr.Textbox(label="Operation result")
            save_button.click(handle_save, inputs=filename_input, outputs=state_output)
            load_button.click(handle_load, inputs=filename_input, outputs=state_output)

    return interface

if __name__ == "__main__":
    dynamic_ai = DynamicAI(vocab_size=50000, embed_dim=256, latent_dim=256, output_dim=256, max_depth=7)
    iface = create_gradio_interface(dynamic_ai)
    iface.launch()