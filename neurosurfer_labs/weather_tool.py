from neurosurfer.tools.tool_spec import ToolSpec, ToolParam, ToolReturn
from neurosurfer.tools.base_tool import BaseTool

class WeatherToolTool(BaseTool):
    spec = ToolSpec(
        name="weather_tool",
        description="Fetches current weather data for a given location",
        when_to_use="When you need to retrieve current weather information",
        inputs=[
            ToolParam(name="location", type="str", description="The location to check the weather for", required=True)
        ],
        outputs=[
            ToolReturn(name="weather_data", type="dict", description="Dictionary containing current weather data")
        ]
    )

    def __call__(self, location: str) -> dict:
        if not isinstance(location, str):
            raise ValueError("Location must be a string")
        
        # Simulate fetching weather data (deterministic, no external calls)
        weather_data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%",
            "wind_speed": "10 km/h"
        }
        
        return weather_data