from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END



load_dotenv()


class State(TypedDict):
    topic: str
    outline: List[str]
    refined_outline: List[str]
    blog_content: str


def generate_outline(state: State):

    topic = state["topic"]

    model = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    parser = StrOutputParser()
    outline_chain = model | parser

    system_message_content = """
    You are an expert outline writer. Your job is to generate a structured outline
    for a Blog post with a section title and key points based on the provided topic.
    """

    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=f"Generate the outline for the topic: {topic}")
    ]
    raw_outline = outline_chain.invoke(messages)
    outline_list = [item.strip() for item in raw_outline.split("\n") if item.strip()]

    return {"outline": outline_list}



def refine_outline(state: State):

    current_outline = state["outline"]
    
    model = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    parser = StrOutputParser()
    outline_chain = model | parser

    message_template  = """
        # Overview
        You are an expert blog outline evaluator and reviser.
        Your task is to take the provided outline and refine it to meet high-quality blog standards.

        # Criteria for Revision:
        (1) Engaging Introduction: Ensure the first section grabs the reader's attention.
        (2) Clear Section Breakdown: Each main point should be distinct and logically organized.
        (3) Logical Flow: The outline should transition smoothly from one section to the next.
        (4) Conclusion with Key Takeaways: The final section should summarize main points and provide clear takeaways.
        (5) Consistency: Ensure formatting and tone are consistent throughout.
        (6) Actionable Titles: Section titles should be clear and indicative of content.

        # Input Outline:
        {outline_text}

        # Output:
        Provide ONLY the revised and improved outline as a list of points. Do not include any
        introductory or concluding remarks outside the outline itself.
        Format each point on a new line.
        """
    
    formatted_outline_content = "\n".join([f"{i+1}. {item}"for i, item in enumerate(current_outline)])

    system_message = message_template.format(outline_text=formatted_outline_content)
    messages = [
        SystemMessage(content=system_message)
    ]
    raw_refined_outline_content = outline_chain.invoke(messages)
    refined_outline_content = [item.strip() for item in raw_refined_outline_content.split("\n") if item.strip()]

    return {"refined_outline": refined_outline_content}




def generate_blog(state: State):

    refined_outline = state["refined_outline"]

    model = ChatGroq(model="llama3-8b-8192", temperature=0.7)
    parser = StrOutputParser()
    outline_chain = model | parser

    prompt_template = """
        You are an expert Blog writer. Your task is to generate a detailed, engaging,
        and well-structured blog post based on the provided outline.

        Ensure the blog post includes:
        - An engaging and relevant title.
        - A compelling introduction that hooks the reader.
        - Well-developed paragraphs for each section of the outline, providing insightful
          information and maintaining a coherent flow.
        - Smooth transitions between different sections.
        - A strong conclusion that summarizes key takeaways and potentially offers
          a final thought or call to action.
        - Use Markdown for formatting (e.g., headings for sections, bold text for emphasis,
          bullet points or numbered lists where appropriate).
        - The tone should be informative and engaging, suitable for a general audience.
        """
    
    formatted_refined_outline = "\n".join([f"{i+1}. {item}" for i, item in enumerate(refined_outline)])

    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(content=f"Here is the refined outline for the blog post:\n{formatted_refined_outline}")
    ]
    generated_blog = outline_chain.invoke(messages)

    return {"blog_content": generated_blog}



builder = StateGraph(State)

builder.add_node("generate_outline", generate_outline)
builder.add_node("refine_outline", refine_outline)
builder.add_node("generate_blog", generate_blog)

builder.add_edge(START, "generate_outline")
builder.add_edge("generate_outline", "refine_outline")
builder.add_edge("refine_outline", "generate_blog")
builder.add_edge("generate_blog", END)

graph = builder.compile()

# user_input = st.text_input("Enter topic name here")

# query_input = {"topic": user_input}

# if query_input:
#     if st.button("click here"):
#         with st.spinner("Getting response..."):
#             response = graph.invoke(query_input)
#             if response:
#                 st.write(response["blog_content"])