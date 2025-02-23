from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from search_jina import searchJina, readerJina
from qa_agent import QAAgent

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class DecisionAssistant:
    def __init__(self):
        # Initialize LLM with streaming
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.5,  # Increased temperature for more creative responses
            max_tokens=5000,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        
        # Initialize QA Agent
        self.qa_agent = QAAgent()
        
        # Create a planning chain using RunnableSequence instead of LLMChain
        planning_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Break down this complex query into detailed steps and determine which tools to use for each step:
            Query: {query}
            Tools: QA agent for RAG, Search for searching web, Reader is there is any URL in the query.
            
            Think about this step by step in detail:
            1. What are the main components and sub-components of this query?
            2. Which tools would be best for each component and why?
            3. How should the results be combined for a comprehensive answer?
            4. What additional context or information might be helpful?
            5. How can we ensure the response is detailed and thorough?
            
            Provide your plan in a clear, detailed format with explanations for each decision."""
        )
        self.planning_chain = planning_prompt | self.llm
        
        # Initialize tools with better descriptions
        self.tools = [
            Tool(
                name="Search",
                func=searchJina,
                description="For comprehensive web searches. Use for current events, facts, or general information. Provide detailed search queries for better results. Input: search query"
            ),
            Tool(
                name="Reader",
                func=readerJina,
                description="For in-depth webpage content analysis. ONLY use with valid URLs starting with http:// or https://. Extracts and analyzes the full content. Input: URL"
            ),
            Tool(
                name="QA",
                func=self.qa_agent.query,
                description="For detailed analysis of loaded documents. Use for specific document queries, ensuring comprehensive understanding and explanation. Input: question"
            )
        ]
        
        # Initialize the agent with better configuration
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3,  # Increased iterations for more thorough analysis
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
    
    def __del__(self):
        """Cleanup when the assistant is destroyed"""
        if hasattr(self, 'qa_agent'):
            self.qa_agent.cleanup()
    
    def process_complex_query(self, query: str) -> str:
        """Handle complex queries that might need multiple tools."""
        try:
            # First, plan the approach
            plan = self.planning_chain.invoke({"query": query})
            if hasattr(plan, 'content'):  # Handle ChatMessage output
                plan = plan.content
            
            # Add the plan to the query for better context
            enhanced_query = f"""Plan: {plan}
            
            Based on this plan, please provide a detailed and comprehensive answer to the following query: {query}
            
            Requirements for your response:
            1. Follow the plan step by step, providing detailed information at each step
            2. Use appropriate tools for each step, extracting maximum relevant information
            3. Combine the results into a coherent, well-structured response
            4. Include specific examples, data points, or quotes when available
            5. Provide context and explanations for your findings
            6. Ensure the response is thorough and addresses all aspects of the query
            7. Minimum response length should be 200 words
            8. Break down complex information into clear, understandable sections
            9. If there are any uncertainties or limitations, explain them clearly
            10. Conclude with a summary of the key points"""
            
            # Execute the enhanced query
            return self.agent.run(enhanced_query)
            
        except Exception as e:
            return f"Error: {str(e)}. Please try rephrasing your question."
    
    def run(self, query: str) -> str:
        """Main entry point for queries."""
        try:
            max_query_length = 500  # Increased max query length
            if len(query) > max_query_length:
                query = query[:max_query_length] + "..."
            
            # Enhanced query wrapper for single tool queries
            enhanced_query = f"""Please provide a detailed and comprehensive answer to this query: {query}

            Requirements for your response:
            1. Be thorough and detailed in your explanation
            2. Include specific examples or evidence when available
            3. Provide context and background information
            4. Ensure a minimum response length of 150 words
            5. Break down complex information into clear sections
            6. Address all aspects of the query
            7. If relevant, include any limitations or uncertainties
            8. End with a clear conclusion or summary

            Take your time to analyze and respond thoughtfully."""
            
            if ('and' in query.lower() or '?' in query[query.find('?')+1:]):
                return self.process_complex_query(query)
            else:
                return self.agent.run(enhanced_query)
        except Exception as e:
            return f"Error processing query: {str(e)}. Please try a shorter or simpler query."

# Example usage
if __name__ == "__main__":
    assistant = DecisionAssistant()
    try:
        # Example complex query
        complex_query = "What is this document about?"
        result = assistant.run(complex_query)
        print(result)
    finally:
        # Ensure cleanup happens when the program exits
        assistant.__del__()