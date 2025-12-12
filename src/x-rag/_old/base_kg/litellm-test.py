import os
import dspy
import asyncio
from dotenv import load_dotenv

from pathlib import Path
SRC = Path(__file__).resolve().parents[2]

class WeatherMan(dspy.Signature):
    """
    You are a weatherman. you recieve a weather status and must generate a weather report.
    """
    input_weather:str = dspy.InputField()
    output_report:str = dspy.OutputField()


async def get_weather_report(llm):
    with dspy.context(lm=llm):
        res = await dspy.Predict(WeatherMan).acall(input_weather="Temp: 6 degrees Celcius. Rain shower. Windspeed: 5km/h. AQI: 4")

    print(f"result: {res.output_report}")


if __name__ == "__main__" :
    load_dotenv(f"{SRC}/secrets.env")
    if os.environ["AWS_ACCESS_KEY_ID"] is None:
        raise Exception("AWS_ACCESS_KEY_ID not found in environment")
    
    llm = dspy.LM("bedrock/us.amazon.nova-pro-v1:0")
    asyncio.run(get_weather_report(llm))