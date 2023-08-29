"""
Conversation prompt templates adapted from FastChat
"""

import dataclasses
from typing import List, Dict


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    name: str
    system: str
    roles: List[str]
    messages: List[List[str]]
    stop_str: str = None
    stop_token_ids: List[int] = None


    def get_prompt(self):

        if "gpt" in self.name and "4all" not in self.name:
            return self.get_prompt_openai()
        else:
            return self.get_prompt_other()

    def get_prompt_other(self) -> str:
        """Get the prompt for generation."""

        ret = ""
        if self.system:
            if len(self.roles) == 3:
                self.system = self.roles[2] + "\n" + self.system
            ret = self.system +  "\n"
        for role, message in self.messages:
            if message:
                ret += role + ":\n" + message + "\n"
            else:
                ret += role + ":\n"
        return ret


    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def get_prompt_openai(self) -> list:
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[0:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        
        if ret[-1]['role'] == self.roles[1]:
            ret = ret[:-1]

        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
        }

    def clear_messages(self):
        self.messages = []


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    name = name.lower()
    for i in conv_templates.keys():
        if i in name:
            return conv_templates[i].copy()
        
    return conv_templates["other"].copy()
    #return conv_templates[name].copy()


 
register_conv_template(
    Conversation(
        name="gpt4all",
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        messages=(),
    )
)
"""
messages = [{"role": "user", "content": "Name 3 colors"}]
"""


register_conv_template(
    Conversation(
        name="gpt",
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        messages=(),
    )
)
"""
  messages=[
        {"role": "system", "content": "You are a Blockchain Development Tutor. Your mission is to guide users from zero knowledge to understanding the fundamentals of blockchain technology and building basic blockchain projects. Start by explaining the core concepts and principles of blockchain, and then help users apply that knowledge to develop simple applications or smart contracts. Be patient, clear, and thorough in your explanations, and adapt to the user's knowledge and pace of learning."},
        {"role": "user", "content": "I'm new to blockchain technology. Can you help me understand what it is and how it works?"}
    ],
"""


register_conv_template(
    Conversation(
        name="wizard-vicuna",
        system="""""",
        roles=("USER", "ASSISTANT"),
        messages=(),
    )
)
"""
USER: What is 4x8?
ASSISTANT:
"""


register_conv_template(
    Conversation(
        name="stable-vicuna",
        system="""""",
        roles=("### Human", "### Assistant"),
        messages=(),
    )
)
"""
### Human: your prompt here
### Assistant:
"""


# register_conv_template(
#     Conversation(
#         name="falcon",
#         system="""""",
#         roles=("User", "Assistant"),
#         messages=(),
#     )
# )
# """   
# Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.
# Daniel: Hello, Girafatron!
# Girafatron:
# """


register_conv_template(
    Conversation(
        name="mpt",
        system="""Below is an instruction that describes a task. Write a response that appropriately completes the request.""",
        roles=("### Instruction", "### Response"),
        messages=(),
        stop_token_ids=[50278, 0],
    )
)

"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week? Explain before answering.
### Response:
"""


register_conv_template(
    Conversation(
        name="nous",
        system="""""",
        roles=("### Input", "### Response", "### Instruction:"),
        messages=(),
        stop_token_ids=[50278, 0],
    )
)

"""
### Instruction:

### Input:

### Response:
"""



