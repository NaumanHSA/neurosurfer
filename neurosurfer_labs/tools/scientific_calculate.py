from neurosurfer.tools.base_tool import BaseTool, ToolResponse
from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn


class ScientificCalculateTool(BaseTool):
    spec = ToolSpec(
        name="scientific_calculate",
        description="Perform scientific calculations such as arithmetic operations, trigonometric functions, logarithms, and exponents.",
        when_to_use="Use this tool when you need to perform complex scientific calculations.",
        inputs=[
            ToolParam(name="expression", type=str, description="The mathematical expression to evaluate.", required=True)
        ],
        returns=ToolReturn(name="result", type=float, description="The result of the evaluated expression.")
    )

    def __call__(self, expression: str):
        try:
            # Evaluate the expression using Python's eval with math functions
            # Replace sin, cos, tan with math.sin, math.cos, math.tan
            expression = expression.replace("sin", "math.sin").replace("cos", "math.cos").replace("tan", "math.tan")
            # Replace log with math.log (natural log)
            expression = expression.replace("log", "math.log")
            # Replace ln with math.log (natural log)
            expression = expression.replace("ln", "math.log")
            # Replace sqrt with math.sqrt
            expression = expression.replace("sqrt", "math.sqrt")
            # Replace pi with math.pi
            expression = expression.replace("pi", "math.pi")
            # Replace e with math.e
            expression = expression.replace("e", "math.e")
            # Evaluate the expression
            result = eval(expression)
            return ToolResponse(
                final_answer=True,
                results=str(result),
                extras={"expression": expression, "result": result}
            )
        except Exception as e:
            return ToolResponse(
                final_answer=False,
                results=f"Error evaluating expression: {str(e)}",
                extras={"expression": expression}
            )