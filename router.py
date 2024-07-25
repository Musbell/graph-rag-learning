import os
from routellm.controller import Controller

os.environ["OPENAI_API_KEY"] = ""
os.environ["ANYSCALE_API_KEY"] = ""

client = Controller(
    # List of routers to initialize
    routers=["mf"],
    # The pair of strong and weak models to route to
    strong_model="",
    weak_model="",
    # The config for the router (best-performing config by default)
    config = {
        "mf": {
            "checkpoint_path": "routellm/mf_gpt4_augmented"
        }
    },
    # Override API base and key for LLM calls
    api_base=None,
    api_key=None,
    # Display a progress bar for operations
    progress_bar=False,
)

response = client.chat.completions.create(
    # This tells RouteLLM to use the MF router with a cost threshold of 0.11593
    model="router-mf-0.11593",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0]["message"]["content"])

routed_model = client.route(
    prompt="What's the squareroot of 144?",
    router="mf",
    threshold=0.11593,
)
print(f"Prompt should be routed to {routed_model}")

import pandas as pd

prompts = pd.Series(["What's the squareroot of 144?", "Who's the last president of the US?", "Is the sun a star?"])
win_rates = client.batch_calculate_win_rate(prompts=prompts, router="mf")

print(f"Calculated win rate for prompts:\n{win_rates.describe()}")