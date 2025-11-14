from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.tools.base_tool import BaseTool

class WeatherToolTool(BaseTool):
    spec = ToolSpec(
        name="weather_tool",
        description="Fetches current weather data for a given location",
        when_to_use="Use this tool when you need to retrieve current weather information for a specific location.",
        inputs=[
            ToolParam(name="location", type="str", description="The location for which to fetch the weather data", required=True)
        ],
        outputs=ToolReturn(
            final_answer=False,
            results="Weather data for the specified location",
            extras={}
        )
    )

    def __call__(self, location: str) -> ToolReturn:
        if not isinstance(location, str):
            raise ValueError("Location must be a string")
        
        # Simulate fetching weather data (deterministic, no external dependencies)
        weather_data = {
            "temperature": "22°C",
            "humidity": "65%",
            "conditions": "Partly Cloudy",
            "location": location
        }
        
        return ToolReturn(
            final_answer=False,
            results=weather_data,
            extras={}
        )