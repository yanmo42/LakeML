import asyncio
import time
from agents import GeneratorAgent, VerifierAgent, SynthesizerAgent, RLGeneratorAgent
from metrics import MetricsCollector

# Shared context: a dictionary to hold messages and iteration info.
shared_context = {
    "messages": [],
    "iteration": 0
}

# Asynchronous function to run an agent's action.
async def async_act(agent, context):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, agent.act, context)

# Run one iteration by having all agents act concurrently.
async def run_iteration(agents, context):
    tasks = [async_act(agent, context) for agent in agents]
    await asyncio.gather(*tasks)

# Main loop: iterate over a number of iterations.
async def main_loop(agents, context, num_iterations, metrics):
    for iteration in range(num_iterations):
        context["iteration"] = iteration
        print(f"\n--- Iteration {iteration} ---")
        await run_iteration(agents, context)
        # Update Q-values for RL agents after each iteration.
        for agent in agents:
            if hasattr(agent, 'update_q_values'):
                agent.update_q_values(context)
        metrics.update(context)
        # Print all messages in the shared context.
        for msg in context["messages"]:
            print(f"{msg['agent_id']} [{msg['type']}]: {msg['content']}")
        # Optionally clear messages between iterations (uncomment next line if desired):
        # context["messages"].clear()
        await asyncio.sleep(1)
    metrics.report()
    # Report Q-tables for any RL agents.
    for agent in agents:
        if hasattr(agent, 'report_q_table'):
            agent.report_q_table()

if __name__ == "__main__":
    # Instantiate agents.
    agents = [
        RLGeneratorAgent("rl_generator_1"),
        GeneratorAgent("generator_2"),
        VerifierAgent("verifier_1"),
        SynthesizerAgent("synthesizer_1")
    ]
    
    metrics = MetricsCollector()
    NUM_ITERATIONS = 10
    
    # Run the main asynchronous loop.
    asyncio.run(main_loop(agents, shared_context, NUM_ITERATIONS, metrics))
