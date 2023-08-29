
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

from llm_classifier.utils.tasks import task_registry
from llm_classifier.utils.conversation import get_conv_template


class Prompter:
    def __init__(self, model: str, task: str, examples: str, max_length: int) -> None:

        self.conv = get_conv_template(model)
        self.task = task_registry[task]
        self.examples = examples
        self.max_length = max_length

    def make_prompt(self, text):

        if self.conv.name in ["gpt4all", "gpt", "nous"]:

            if self.examples == "zero":
                self.conv.system = self.task.prefix.format(categories=self.task.categories)
                self.conv.append_message(role= self.conv.roles[0], message=text)

            elif self.examples == "few":

                for i, _ in enumerate(self.task.examples):
                    self.task.examples[i]["user"] = self.conv.roles[0]
                    self.task.examples[i]["assistant"] = self.conv.roles[1]

                prompt = PromptTemplate(
                    input_variables=["user", "text", "assistant", "response"],
                    template="""{user}: {text}\n\n{assistant}: {response}""",
                )

                example_selector = LengthBasedExampleSelector(
                        examples=self.task.examples,
                        example_prompt=prompt, 
                        max_length=self.max_length,
                    )
                                
                prompt = FewShotPromptTemplate(
                        example_selector=example_selector,
                        example_prompt=prompt,
                        prefix=self.task.prefix.format(categories=self.task.categories),
                        input_variables=["user","text"],
                        suffix="{user}: {text}",
                        example_separator="\n\n",
                    )
                
                self.conv.system = prompt.format(user=self.conv.roles[0], text=text)


        else:

            if self.examples == "zero":
                self.conv.append_message(role= self.conv.roles[0], message=self.task.prefix.format(categories=self.task.categories))
                self.conv.append_message(role= self.conv.roles[0], message=text)


            elif self.examples == "few":


                for i, _ in enumerate(self.task.examples):
                    self.task.examples[i]["user"] = self.conv.roles[0]
                    self.task.examples[i]["assistant"] = self.conv.roles[1]

                prompt = PromptTemplate(
                    input_variables=["user", "text", "assistant", "response"],
                    template="""{user}: {text}\n\n{assistant}: {response}""",
                )

                example_selector = LengthBasedExampleSelector(
                    examples=self.task.examples,
                    example_prompt=prompt, 
                    max_length=self.max_length/2,
                )
                            
                self.prompt = FewShotPromptTemplate(
                    example_selector=example_selector,
                    example_prompt=prompt,
                    prefix=self.task.prefix.format(categories=self.task.categories),
                    input_variables=["user","text"],
                    suffix="{user}: {text}",
                    example_separator="\n\n",
                )

                prompt = self.prompt.format(user=self.conv.roles[0], text=text)       
                self.conv.append_message(role= self.conv.roles[0], message=prompt)
        
        
        prompt = self.conv.get_prompt()

        return prompt

    def parse(self, text):
        return text.strip().lower()

    def truncate(self, prompt: str, text: str,):

        if len(prompt) > self.max_length:

            remaining_length = self.max_length - (len(prompt) - len(text))

            # Take from beginning and end
            text = text[:int(0.5 * remaining_length)] + text[-int(0.5 * remaining_length):]
        
            return self.make_prompt(text)
        
        else:
            return prompt