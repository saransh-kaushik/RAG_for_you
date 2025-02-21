from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
import os
from search_jina import searchJina, readerJina
from qa_agent import rag_chain

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class DecisionAssistant:
    def __init__(self):
        # Initialize LLM with streaming
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=2000,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        
        # Create a planning chain using RunnableSequence instead of LLMChain
        planning_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Break down this complex query into steps and determine which tools to use for each step:
            Query: {query}
            
            Think about this step by step:
            1. What are the main components of this query?
            2. Which tools would be best for each component?
            3. How should the results be combined?
            
            Provide your plan in a clear format."""
        )
        self.planning_chain = planning_prompt | self.llm
        
        # Initialize tools with better descriptions
        self.tools = [
            Tool(
                name="Search",
                func=searchJina,
                description="For general web searches. Use for current events, facts, or general information. Input: search query"
            ),
            Tool(
                name="Reader",
                func=readerJina,
                description="For reading webpage content. ONLY use with valid URLs starting with http:// or https://. Input: URL"
            ),
            Tool(
                name="QA",
                func=rag_chain.invoke,
                description="For questions about loaded documents (like resumes). Use for specific document queries. Input: question"
            )
        ]
        
        # Initialize the agent with better configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=2,  # Allow more iterations for complex queries
            early_stopping_method="generate",
            handle_parsing_errors=True  # Better error handling
        )
    
    def process_complex_query(self, query: str) -> str:
        """Handle complex queries that might need multiple tools."""
        try:
            # First, plan the approach
            plan = self.planning_chain.invoke({"query": query})
            if hasattr(plan, 'content'):  # Handle ChatMessage output
                plan = plan.content
            
            # Add the plan to the query for better context
            enhanced_query = f"""Plan: {plan}
            
            Based on this plan, please answer the following query: {query}
            
            Remember to:
            1. Follow the plan step by step
            2. Use appropriate tools for each step
            3. Combine the results coherently"""
            
            # Execute the enhanced query
            return self.agent.run(enhanced_query)
            
        except Exception as e:
            return f"Error: {str(e)}. Please try rephrasing your question."
    
    def run(self, query: str) -> str:
        """Main entry point for queries."""
        # Truncate long queries
        max_query_length = 200
        if len(query) > max_query_length:
            query = query[:max_query_length] + "..."
        
        try:
            # Check if query might be complex (contains multiple questions or 'and')
            if ('and' in query.lower() or '?' in query[query.find('?')+1:]):
                return self.process_complex_query(query)
            else:
                return self.agent.run(query)
        except Exception as e:
            return f"Error processing query: {str(e)}. Please try a shorter or simpler query."

# Example usage
if __name__ == "__main__":
    assistant = DecisionAssistant()
    
    # Example complex query
    complex_query = "What is the name and city of the candidate in the resume?"
    result = assistant.run(complex_query)
    print(result)