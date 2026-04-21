from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , scrape_url 
from dotenv import load_dotenv

load_dotenv()

#model setup 
llm = ChatMistralAI(model="mistral-small-2506", temperature = 0)

#1st agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search]
    )

#2nd agent 
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url]
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research analyst who transforms fragmented information into clear, structured insight."),
    ("human", """Write a detailed research report on the topic below.

        Topic: {topic}

        Collected Material:
        {research}

        Write a well-structured report that:
        - introduces the topic with necessary context
        - synthesizes the most important insights (group related ideas, avoid repetition)
        - highlights implications or trends where relevant
        - concludes with a concise takeaway

        End with a 'Sources' section listing all referenced URLs."""),
        ])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are reviewing a research report for quality, clarity, and depth. Be honest and specific."),
    ("human", """Review the research report below and evaluate it.

        Report:
        {report}

        Provide:

        Score: X/100

        What works well in 2 lines:
        - ...

        What could be improved in 2 lines:
        - ...

        Overall assessment (1–2 lines):
        ...
        """)
        ])
critic_chain = critic_prompt | llm | StrOutputParser()
