from pydantic import BaseModel, Field
from utils import image_to_base64, call_llm
import backoff
import orjson

class ReflectionResult(BaseModel):
    """Represents the assessment of the previous step."""
    last_step_correct: bool = Field(description="Whether the previous step was correct.")
    last_step_redundant: bool = Field(description="Whether the previous step was redundant.")
    reflection: str = Field(description="Reflection on the previous step's outcome and the current state.")


TRANSIT_REFLECTION_WITH_IMAGE_PATCH_FORMAT_PROMPT = """You are an judge of a computer-use agent. You will be given a task, the agent's history actions, agent last action and thought process with 2 full screenshots and 1 image patch of the first screenshot.
- Thought is the reasoning for the history steps and prediction for the next step.
- Action is the summary of the code
- Code is the code that will be executed.
- Note:
    - If there is mouse related code that need coordinate, the center of the red circle in the screenshot shows the position. But do not mention the red circle or red dot in any part of your response.
    - You will be provided with 3 images. The first image is the full screenshot of the observation of the last action. The second image is a cropped image from the first image centered around the coordinate of the action. The thrid image is the computer's full screenshot after executing the last action (code).
    - The image patch is zoomed in on the area where the action was performed. You should mainly focus on the differences between the full screenshots before and after the action, but also consider the image patch when the action is related to small buttons or elements, for example when clicking new tab button, minimize button, close button, or other buttons.
    - If there is only one screenshot, you can say what is expected to change.

# Task:
{goal}

# History steps:
{history_steps}

# Last step:
## Thought:
{thought}
## Action:
{action}
## Code:
{code}

Your response should include 3 parts:
1. Is the last step redundant:
    - If the last step is doing unnecessary action or action that is not related to the task, for example, clicking irrelevant places, open irrelevant applications, or unnecessary scrolls, you should mark it as redundant.
    - If the last step is a repeat of the former step, you should mark it as redundant.
    - Too many scrolls or drags of the scroll bar, or too many clicks of the same button, or too many clicks of the same element, you should mark it as redundant.

2. Is the last step incorrect:
    - If the action is related to the task but executing the code did not produce the expected change, you should mark it as incorrect.
    - If the action and the code do not align, you should mark it as incorrect. For example the action tries to click an element but failed according to the screenshot.
    - The last screenshot shows the application or window is not fully loaded, but the code is executed.
    - If there is any mistake in the thought or action.
    - You should carefully examine the click/drag related actions. In many cases, the action want's to click a target, but it doesn't match the element at the center of the red circle in the screenshot.
        - If the action doesn't matches the element at the center of the red circle, the action is incorrect.
        - If the action didn't bring the expected change, the action is incorrect.
        - The action may say click close (X) button, but it may be clicking the minimize button or new tab button, or other buttons, then it would be incorrect. Be careful about the action and the code.

3. Reflection:
    - You should first provide a natural summary of the visual changes between the last screenshot and the current screenshot. If there is no change, please mention it.
    - If the last step is correct and not redundant, you should then say the step is necessary and how it is effective.
    - If the last step is incorrect, you should then provide a clear explanation of the error.
    - If the last step is redundant, you should then provide a clear explanation.
"""

TRANSIT_REFLECTION_FORMAT_PROMPT = """You are an judge of a computer-use agent. You will be given a task, the agent's history actions, agent last action and thought process with 2 screenshots.
- Thought is the reasoning for the history steps and prediction for the next step.
- Action is the summary of the code
- Code is the code that will be executed.
- The first screenshot is the observation of the last action and the second image is the computer state after executing the last action (code).

# Task:
{goal}

# History steps:
{history_steps}

# Last step:
## Thought:
{thought}
## Action:
{action}
## Code:
{code}

Your response should include 3 parts:
1. Is the last step redundant:
    - If the last step is doing unnecessary action or action that is not related to the task, for example, clicking irrelevant places, open irrelevant applications, or unnecessary scrolls, you should mark it as redundant.

2. Is the last step incorrect:
    - If the action is related to the task but executing the code did not produce the expected change, you should mark it as incorrect.
    - If the action and the code do not align, you should mark it as incorrect. For example the action tries to click an element but failed according to the screenshot.
    - The last screenshot shows the application or window is not fully loaded, but the code is executed.
    - If there is any mistake in the thought action.

3. Reflection:
    - You should first provide a natural summary of the visual changes between the last screenshot and the current screenshot. If there is no change, please mention it.
    - If the last step is correct and not redundant, you should then say the step is necessary and how it is effective.
    - If the last step is incorrect, you should then provide a clear explanation of the error.
    - If the last step is redundant, you should then provide a clear explanation.
"""

TRANSIT_REFLECTION_FORMAT_PROMPT_TERMINATE_STEP = """You are an judge of a computer-use agent. You will be given a task, the agent's history actions, agent last action and thought process with 1 screenshot.
- Thought is the reasoning for the history steps and prediction for the next step.
- Action is the summary of the code
- Code is the code that will be executed.

# Task:
{goal}

# History steps:
{history_steps}

# Last step:
## Thought:
{thought}
## Action:
{action}
## Code:
{code}

Your response should include 3 parts:
1. Is the last step redundant:
    - The terminate action is always not redundant.

2. Is the last step incorrect:
    - The input step will be the last step of the agent, so it will be a terminal action including 'success' or 'fail'.
    - If the code is success but the screenshot is not related at all to the task, you should mark it as incorrect.
    - If the code is success accroding to the history and the screenshot, the task is not completed, you should mark it as incorrect.
    - If the code is fail, according to the history, the task won't be able to be completed, you should mark it as correct.
    - You should judge according to the history and the screenshot whether the action is correct.
    - If there is any mistake in the thought action.

3. Reflection:
    - If the last step is correct, you should explain why the step.
    - If the last step is incorrect, you should then provide a clear explanation of the error.
"""

REFLECTION_FORMAT_PROMPT = """YOUR RESPONSE MUST BE EXACTLY ONE VALID JSON OBJECT. NO MARKDOWN, NO EXTRA TEXT.

Here is the exact JSON structure you must follow:

{
    "last_step_correct": bool,  // true or false
    "last_step_redundant": bool,  // true or false
    "reflection": str
}
"""

def build_reflection_messages(
    task: str,
    history_steps: str,
    current_step: dict,
    current_image,
    image_patch,
    next_image
    ) -> list:

    content = []
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(current_image)}", "detail": "high"}})
    if not next_image:
        print("============================")
        print(f"Image exists: {next_image is not None}", current_step["code"])
        print("=============================")

    prompt_template = TRANSIT_REFLECTION_FORMAT_PROMPT
    if current_step["code"].startswith("computer.terminate"):
        prompt_template = TRANSIT_REFLECTION_FORMAT_PROMPT_TERMINATE_STEP

    if image_patch:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image_patch)}", "detail": "high"}})
        prompt_template = TRANSIT_REFLECTION_WITH_IMAGE_PATCH_FORMAT_PROMPT

    if next_image:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(next_image)}", "detail": "high"}})


    content.append({
        "type": "text",
        "text":
        prompt_template.format(
            goal = task,
            history_steps = history_steps,
            thought = current_step["thought"],
            action = current_step["action"],
            code = current_step["code"]
            )+"\n\n"+REFLECTION_FORMAT_PROMPT}
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
def gen_reflection_thought(
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
    ReflectionResult.model_validate(parsed_data)
    return parsed_data
