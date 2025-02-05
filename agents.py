import time
import uuid
import random

class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def read_context(self, context):
        # For now, simply return the list of messages.
        return context.get("messages", [])

    def process(self, messages):
        # Abstract method: override in subclasses.
        raise NotImplementedError

    def write_context(self, context, message):
        context["messages"].append(message)

    def act(self, context):
        local_messages = self.read_context(context)
        message = self.process(local_messages)
        if message:
            self.write_context(context, message)

# Basic GeneratorAgent: Generates a random proposal.
class GeneratorAgent(Agent):
    def process(self, messages):
        proposal = f"Hypothesis_{random.randint(1, 100)}"
        message = {
            "id": str(uuid.uuid4()),
            "agent_id": self.agent_id,
            "type": "proposal",
            "content": proposal,
            "priority": "normal",
            "timestamp": time.time(),
            "confidence": random.uniform(0.5, 1.0)
        }
        return message

# VerifierAgent: Chooses a proposal and either verifies or rejects it.
class VerifierAgent(Agent):
    def process(self, messages):
        proposals = [msg for msg in messages if msg["type"] == "proposal"]
        if proposals:
            chosen = random.choice(proposals)
            verdict = "Verified" if random.random() > 0.3 else "Rejected"
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": self.agent_id,
                "type": "verification",
                "content": f"{verdict}: {chosen['content']}",
                "priority": "high",
                "timestamp": time.time(),
                "confidence": random.uniform(0.5, 1.0),
                "ref_id": chosen["id"]  # Link back to the proposal.
            }
            return message
        return None

# SynthesizerAgent: Combines proposals into a meta-hypothesis.
class SynthesizerAgent(Agent):
    def process(self, messages):
        proposals = [msg["content"] for msg in messages if msg["type"] == "proposal"]
        if len(proposals) >= 2:
            combined = f"MetaHypothesis: ({random.choice(proposals)}) + ({random.choice(proposals)})"
            message = {
                "id": str(uuid.uuid4()),
                "agent_id": self.agent_id,
                "type": "synthesis",
                "content": combined,
                "priority": "normal",
                "timestamp": time.time(),
                "confidence": random.uniform(0.6, 1.0)
            }
            return message
        return None

# RLGeneratorAgent: Uses Q-learning to decide what type of proposal to generate.
class RLGeneratorAgent(Agent):
    def __init__(self, agent_id, epsilon=0.2, alpha=0.1, gamma=0.9):
        super().__init__(agent_id)
        # Q-table: keys are (state, action) tuples, values are Q-values.
        self.q_table = {}
        # Possible actions representing proposal types.
        self.actions = ["low", "medium", "high"]
        # Mapping from proposal id to (state, action).
        self.last_actions = {}
        self.epsilon = epsilon  # Exploration rate.
        self.alpha = alpha      # Learning rate.
        self.gamma = gamma      # Discount factor.

    def _discretize_state(self, messages):
        # Use the number of proposals as a simple state.
        num_proposals = len([msg for msg in messages if msg["type"] == "proposal"])
        if num_proposals < 3:
            return 0
        elif num_proposals <= 5:
            return 1
        else:
            return 2

    def _choose_action(self, state):
        # ε-greedy action selection.
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Choose the action with the highest Q-value for this state.
            q_values = {action: self.q_table.get((state, action), 0) for action in self.actions}
            return max(q_values, key=q_values.get)

    def process(self, messages):
        state = self._discretize_state(messages)
        action = self._choose_action(state)
        proposal = f"Hypothesis_{action}_{random.randint(1, 100)}"
        message = {
            "id": str(uuid.uuid4()),
            "agent_id": self.agent_id,
            "type": "proposal",
            "content": proposal,
            "priority": "normal",
            "timestamp": time.time(),
            "confidence": random.uniform(0.5, 1.0)
        }
        # Store the (state, action) pair for later Q-table updates.
        self.last_actions[message["id"]] = (state, action)
        return message

    def update_q_values(self, context):
        # Look for verification messages that reference proposals from this agent.
        for msg in context.get("messages", []):
            if msg["type"] == "verification" and "ref_id" in msg:
                ref_id = msg["ref_id"]
                if ref_id in self.last_actions:
                    (state, action) = self.last_actions[ref_id]
                    # Reward: +1 for Verified, -1 for Rejected.
                    reward = 1 if "Verified" in msg["content"] else -1
                    current_q = self.q_table.get((state, action), 0)
                    max_future = max([self.q_table.get((state, a), 0) for a in self.actions])
                    new_q = current_q + self.alpha * (reward + self.gamma * max_future - current_q)
                    self.q_table[(state, action)] = new_q
                    # Remove the entry so that the same proposal is not updated again.
                    del self.last_actions[ref_id]

    def report_q_table(self):
        print(f"\n[{self.agent_id}] Q-table:")
        for key, value in self.q_table.items():
            print(f"  State {key[0]}, Action {key[1]}: Q-value = {value:.3f}")
