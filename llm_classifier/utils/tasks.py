
import dataclasses
from typing import List, Dict


@dataclasses.dataclass
class Task():
    """A class that keeps Prompt Elements by Task"""
    
    name: str
    categories: List[str]
    examples: List[dict]
    prefix: str


task_registry: Dict[str, Task] = {}


def register_task(template: Task, override: bool = False):
    """Register a new task template."""
    if not override:
        assert template.name not in task_registry, f"{template.name} has been registered."
    task_registry[template.name] = template


def get_task_template(name: str) -> Task:
    """Get a task template."""
    return task_registry[name].copy()



register_task( 
    Task(
        name="topic",
        categories=  [
                "government/politics", 
                "sports/fitness", 
                "business/economics", 
                "arts/culture/entertainment", 
                "crime/public safety", 
                "school/education",
                "miscellaneous",
            ],
        examples= [
                {
                "text": "House Bill 999 would ban critical race theory, majors in women's or gender studies and funding for diversity, equity and inclusion programs.", 
                 "response": "government/politics",
                 },
                {
                "text": "“Wear Sunscreen” Hugh Jackman urges sun safety after skin cancer tests https://www.britishherald.com/2023/04/04/wear-sunscreen-hugh-jackman-urges-sun-safety-after-skin-cancer-tests/ #BritishHerald #news #health #healthnews #skincancer #hughjakman #Sunsafty", 
                 "response": "miscellaneous",
                 },
                {
                "text": "The Houston Rockets have agreed to hire Ime Udoka as their next coach, ending a seven-month exile for the former Boston Celtics coach who was suspended for engaging in an improper relationship with a female team employee.", 
                 "response": "sports/fitness",
                 },
            ],
        prefix= """You are given a post and must assign it one of the following topic names: {categories}. Respond with the topic for the input. No other output form is accepted.""",
    )    
)    


register_task( 
    Task(
        name="partisanship",
        categories=  [
                "liberal", 
                "moderate", 
                "conservative", 
            ],
        examples= [
                {
                "text": """ALERT! Fauci Agency MUTANT VIRUSES Uncovered! READ: https://www.judicialwatch.org/coronavirus-mutants/” | Pray for America! | "All bets are off. You can expect grand jury indictments of leftist politicians like Biden, [former House Speaker Nancy] Pelosi and [Senate Majority Leader Chuck] Schumer as surely as night follows day.""", 
                "response": "conservative",
                 },
                {
                "text": """Dear friends and family members who watch Fox "News," I'Il be graciously accepting apologies whenever you're ready. | Bye, Felicia! Hopefully we never hear from Tucker EVER AGAIN. | DA Alvin Bragg just sued Jim Jordan. Happy Tuesday! 👏👏""", 
                "response": "liberal",
                 },
                {
                "text": "From Hattie McDaniel to Denzel Washington, these Black stars have made their mark on Hollywood! Check out our list of iconic Black celebrities on the Hollywood Walk of Fame. | Happy 62nd Birthday, Eddie Murphy: Here are 10 classic films (plus one bonus) to celebrate his big day. | Texas Southern University, Cheerleaders make history as the first HBCU to win the NCA's competition in 75 five years. 🎉", 
                "response": "moderate",
                 },
            ],
        prefix= """You are given a text that matches one of the following political leanings: {categories}. Respond with the political leaning for the text. No other output form is accepted.""",
    )    
)    
     

register_task( 
    Task(
        name="trustworthy",
        categories=  [
                "False",
                "True",
            ],
        examples= [
                {
                "text": "Most bad-ass move of the year. 😎 🇺🇸 | Country Superstar Punishes Woke Bud Light – Days After Kid Rock Smackdown, Tritt Straight Out Deletes Them | Flashback: Hollweird's collapse began right HERE.", 
                 "response": "False",
                 },
                {
                "text": "SpaceX successfully launched its next-generation Starship cruise vessel for the first time in an uncrewed test flight that ended minutes later with the vehicle exploding in the sky. https://yhoo.it/3L0OktA | More cheese. More onions. More sauce. 🍔 https://yhoo.it/3oiBnUe | SpaceX's two-stage rocket ship blasted off from the company's Starbase spaceport Thursday morning. Less than four minutes into the flight, it exploded. https://yhoo.it/3owbLDr", 
                 "response": "True",
                 },
                {
                "text": "A 10-foot female great white shark washed ashore near 10th Avenue North in North Myrtle Beach Wednesday night. | Gigi the giraffe is pregnant and will deliver her calf in the coming weeks. The excitement is giraffing us crazy! | The El Paso Zoo announced the birth of a male giraffe. 🦒 This marks the first time that a giraffe has been born at the El Paso Zoo in the city’s history.", 
                 "response": "True",
                 },
            ],
        prefix= """You are given an input and must respond with {categories[1]} if the input is trustworth and {categories[0]} if it is not. Trustworthy is defined as text that is factually correct and not sensational. Only respond with a value in {categories}. No other output form is accepted.""",
    )    
)   


register_task( 
    Task(
        name="partisanship_account",
        categories=  [
                "liberal", 
                "centrist", 
                "conservative", 
            ],
        examples= [
                {
                "text": """The New York Times""", 
                "response": "liberal",
                 },
                {
                "text": """Breitbart""", 
                "response": "conservative",
                 },
                {
                "text": """CNET""", 
                "response": "centrist",
                 },
            ],
        prefix= """You are given the name of a popular news outlet. Respond with the news outlet's political leaning from the list {categories}. No other output form is accepted.""",
    )    
)   


register_task( 
    Task(
        name="trustworthy_account",
        categories=  [
                "False",
                "True",
            ],
        examples= [
                {
                "text": "The New York Times",
                "response": "True",
                 },
                {
                "text": "Breitbart", 
                 "response": "False",
                 },
                {
                "text": "CNET", 
                 "response": "True",
                 },
            ],
        prefix= """You are given the name of a popular news outlet and must respond with {categories[1]} if the news outlet is trustworthy and {categories[0]} if it is not. No other output form is accepted. A news outlet is trustworthy if its content is factually correct and credbible.""",
    )    
)   
