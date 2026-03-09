from pydantic import BaseModel, Field
from utils import image_to_base64, call_llm
import backoff
import orjson

class ReflectionResult(BaseModel):
    """Represents the assessment of the previous step."""
    reflection: str = Field(description="Reflection on the previous step's outcome and the current state.")


TRANSIT_REFLECTION_FORMAT_PROMPT = """You will be given a task, agent last action and thought process with 2 screenshots before and after the last action.
- Thought is the reasoning for the history steps and prediction for the next step.
- Action is the summary of the code
- Code is the code that will be executed.
- Note:
    - If there is mouse related code that need coordinate, the center of the red circle in the screenshot shows the position. But do not mention the red circle or red dot in any part of your response.
    - The first screenshot is the observation of the last action and the second image is the computer state after executing the last action (code).

# Task:
{goal}

# Last step:
## Thought:
{thought}
## Action:
{action}
## Code:
{code}

Your response should include a reflection part to describe the visual changes between the last screenshot and the current screenshot to explain the effectiveness of the last step.

"""

TRANSIT_REFLECTION_FORMAT_PROMPT_LAST_INCORRECT = """You will be given a task, agent last action and thought process with 2 screenshots.
- Thought is the reasoning for the history steps and prediction for the next step.
- Action is the summary of the code
- Code is the code that will be executed.
- The first screenshot is the observation of the last action and the second image is the computer state after executing the last action (code).

# Task:
{goal}

# Last step:
## Thought:
{thought}
## Action:
{action}
## Code:
{code}

The last step is an INCORRECT step. The code did a wrong action. Your response should include a reflection part to explain:
1. What changes you see in the screenshots before and after the last action. If there is no change, explain why the last step is redundant.
2. Why the last step is incorrect.
"""

REFLECTION_ONLY_FORMAT_PROMPT = """YOUR RESPONSE MUST BE EXACTLY ONE VALID JSON OBJECT. NO MARKDOWN, NO EXTRA TEXT.

Here is the exact JSON structure you must follow:

{
    "reflection": str
}
"""

def build_reflection_messages(task: str, history_steps: str, current_step: dict, current_image, image_patch, next_image) -> list:
    content = []
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(current_image)}", "detail": "high"}})
    if next_image:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(next_image)}", "detail": "high"}})

    prompt_template = TRANSIT_REFLECTION_FORMAT_PROMPT if current_step["last_step_correct"] else TRANSIT_REFLECTION_FORMAT_PROMPT_LAST_INCORRECT

    content.append({
        "type": "text",
        "text":
        prompt_template.format(
            goal = task,
            thought = current_step["thought"],
            action = current_step["action"],
            code = current_step["code"]
            )+"\n\n"+REFLECTION_ONLY_FORMAT_PROMPT}
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    return messages

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_time=180,
    max_tries=2,
    jitter=backoff.full_jitter,
)
def gen_reflection_thought_with_prior_judge(
        client,
        model: str,
        goal: str,
        history_steps: str,
        current_step: dict,
        image,
        image_patch=None,
        next_image=None
    ) -> dict:
    """
    Parse a structured LLM response containing JSON code blocks. If the response
    includes a ```json ... ``` fenced block, extract only that content. Otherwise
    parse the entire string.

    Returns a dict that must conform to ReflectionResult's schema.
    """
    reflection_messages = build_reflection_messages(
        task=goal,
        history_steps=history_steps,
        current_step=current_step,
        current_image=image,
        image_patch=image_patch,
        next_image=next_image
    )

    response_str = call_llm(client, messages=reflection_messages, model=model, temperature=0)
    print("\nReflection Thought:")
    print(response_str)

    # If the response contains a ```json block, extract the JSON content
    if "```json" in response_str:
        response_str = response_str.split("```json")[1].split("```")[0].strip()

    parsed_data = orjson.loads(response_str)
    return parsed_data
