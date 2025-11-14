from neurosurfer.tools.tool_spec import ToolSpec, ToolParam
from neurosurfer.tools.base_tool import BaseTool, ToolResponse

class CalculatorTool(BaseTool):
    spec = ToolSpec(
        name="calculator",
        description="Perform basic arithmetic operations such as addition, subtraction, multiplication, and division.",
        when_to_use="Use this tool when you need to perform basic arithmetic operations.",
        inputs=[
            ToolParam(name="num1", type="float", description="The first number.", required=True),
            ToolParam(name="num2", type="float", description="The second number.", required=True),
            ToolParam(name="operation", type="string", description="The operation to perform: 'add', 'subtract', 'multiply', or 'divide'.", required=True)
        ]
    )

    def __call__(self, num1: float, num2: float, operation: str) -> ToolResponse:
        if operation not in ["add", "subtract", "multiply", "divide"]:
            return ToolResponse(
                final_answer=False,
                results="Invalid operation. Supported operations are 'add', 'subtract', 'multiply', and 'divide'.",
                extras={}
            )
        
        if operation == "divide" and num2 == 0:
            return ToolResponse(
                final_answer=False,
                results="Division by zero is not allowed.",
                extras={}
            )
        
        try:
            if operation == "add":
                result = num1 + num2
            elif operation == "subtract":
                result = num1 - num2
            elif operation == "multiply":
                result = num1 * num2
            elif operation == "divide":
                result = num1 / num2
        except Exception as e:
            return ToolResponse(
                final_answer=False,
                results=f"An error occurred: {str(e)}",
                extras={}
            )
        
        return ToolResponse(
            final_answer=True,
            results=result,
            extras={}
        )