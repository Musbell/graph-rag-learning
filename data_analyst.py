from interpreter import interpreter

interpreter.os = True
interpreter.llm.supports_vision = True

interpreter.llm.model = "gpt-4o"


# interpreter.llm.api_base="http://localhost:6060/v1/"
# interpreter.llm.api_key="no_api_key"
interpreter.llm.supports_functions = True
interpreter.llm.context_window = 110000
interpreter.llm.max_tokens = 4096
interpreter.auto_run = True
interpreter.loop = True
interpreter.system_message = ("""
                               ROLE:
                               Agricultural Data Analyst
                               ___
                               GOAL:
                               Analyze historical weather and soil data to identify patterns and trends relevant to crop production.
                               ___
                               BACKSTORY:
                               You are an expert in agricultural data analysis with a deep understanding of how weather and soil conditions affect crop growth. Your insights help farmers make data-driven decisions to optimize their crop production.
                               """)


interpreter.custom_instructions = """
                                TASK DESCRIPTION:
                                Analyze the historical weather and soil data for crop production. Focus on identifying significant patterns and correlations between variables. Pay special attention to how temperature, precipitation, humidity, wind speed, soil temperature, and soil moisture interact and affect potential crop growth. Your final report should clearly articulate the key trends, anomalies, and potential impact on different crop types.
                                ___
                                EXPECTED OUTPUT:
                                A comprehensive 3-paragraph report on the key agricultural data trends and their implications.
                                """


interpreter.chat("""Use this data https://raw.githubusercontent.com/Musbell/gis_data/main/data.csv to provide data 
using the historical data based on these recommendations include precise date and time with relevant data: Agronomic 
Recommendations for Enhanced Tomato Production In analyzing tomato production to provide comprehensive insights and 
advisories for a complete crop calendar, several key data requirements need to be considered. These include soil 
temperature ranges, optimal weather conditions, soil moisture levels, fertilizer application timing and rates, 
pest and disease monitoring, and expected harvest dates. By integrating historical and forecast soil and weather 
data, farmers can make informed decisions to maximize crop yield and sustainability. Soil Temperature Ranges: Seed 
Germination: Soil temperature between 20-30°C is optimal [Data: Reports (13, 8, 16, 1)]. Vegetative Growth: Maintain 
soil temperature between 20-28°C [Data: Reports (11, 14, 3, 15, 0)]. Flowering and Fruit Setting: Soil temperature 
should be 15-25°C [Data: Reports (7, 6, 10)]. Optimal Weather Conditions: Vegetative Growth: Consistent sunlight 
exposure with temperatures of 25-32°C during the day and 15-20°C at night is ideal [Data: Reports (13, 8, 16, 
1)]. Flowering and Fruit Setting: Moderate temperatures around 70-80°F with adequate humidity are favorable [Data: 
Reports (11, 14, 3, 15, 0)]. Soil Moisture Levels and Irrigation Needs: Seedlings: Moist but not waterlogged soil is 
required. Vegetative Growth: Maintain soil moisture at 75-85% field capacity. Flowering and Fruiting Stages: Reduce 
moisture to 60-75% field capacity [Data: Reports (7, 6, 10)]. Fertilizer Application Timing and Rates: Before 
Planting: Apply balanced fertilizer with nitrogen, phosphorus, and potassium. During Growth Stages: Top-dress with 
nitrogen to support growth and fruit development [Data: Reports (13, 8, 16, 1)]. Pest and Disease Monitoring: 
Throughout Growing Season: Regularly scout for pests and diseases like African Bollworm and Fusarium Wilt. Control 
Measures: Implement integrated pest management strategies [Data: Reports (11, 14, 3, 15, 0)]. Expected Harvest Dates: 
Estimation: Harvest typically occurs 60-85 days after transplanting. Factors Influencing Harvest: Disease resistance, 
maturity period, and environmental conditions [Data: Reports (7, 6, 10)]. By utilizing these data points and 
insights, farmers can create a detailed crop calendar tailored to their specific conditions. Monitoring soil 
temperature, weather conditions, moisture levels, and implementing appropriate fertilizer and pest management 
strategies at the right time will contribute to enhanced tomato production, reduced risks, and improved overall crop 
yield and sustainability.""")




