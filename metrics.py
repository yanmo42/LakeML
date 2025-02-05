class MetricsCollector:
    def __init__(self):
        self.hypotheses_history = []      # Track count of proposal messages.
        self.verifications_history = []   # Track count of verification messages.
        self.syntheses_history = []       # Track count of synthesis messages.
    
    def update(self, context):
        messages = context.get("messages", [])
        num_proposals = len([m for m in messages if m["type"] == "proposal"])
        num_verifications = len([m for m in messages if m["type"] == "verification"])
        num_syntheses = len([m for m in messages if m["type"] == "synthesis"])
        self.hypotheses_history.append(num_proposals)
        self.verifications_history.append(num_verifications)
        self.syntheses_history.append(num_syntheses)
    
    def report(self):
        print("\n--- Metrics Report ---")
        print("Proposals per iteration:", self.hypotheses_history)
        print("Verifications per iteration:", self.verifications_history)
        print("Syntheses per iteration:", self.syntheses_history)
