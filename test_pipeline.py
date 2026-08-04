# test_pipeline.py



import asyncio
import re
from pipeline import Pipeline

def run_test(user_prompt: str):
	print("=" * 80)
	print(f"🧪 TESTING PROMPT: {user_prompt}")
	print("=" * 80)

	# 1. Instantiate the Pipeline
	pipeline = Pipeline()

	# (Optional) Override Valve defaults for local testing if needed
	# pipeline.valves.OLLAMA_BASE_URL = "http://darkstar.local:11434"
	pipeline.valves.DB_PATH = "fec_data.db"
	pipeline.valves.MODEL_NAME = "qwen2.5-coder:1.5b"
	# pipeline.valves.MODEL_NAME = "qwen2.5-coder:7b"
	# pipeline.valves.MODEL_NAME = "qwen2.5-coder:32b"
	pipeline.valves.MAX_RETRIES = 3

	# 2. Trigger the startup lifecycle hook
	asyncio.run(pipeline.on_startup())

	# 3. Simulate an Open WebUI payload
	messages = [{"role": "user", "content": user_prompt}]
	body = {"messages": messages}

	try:
		# 4. Call pipe() directly
		response = pipeline.pipe(
			user_message=user_prompt,
			model_id="local-test-model",
			messages=messages,
			body=body
		)

		print("\n--- 🏁 RAW PIPELINE RESPONSE ---")
		print(response)

		# 5. Extract and print specific components for validation
		print("\n--- 🔍 DIAGNOSTICS & VERIFICATION ---")
		
		# Check if SQL was generated and captured
		sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
		if sql_match:
			print(f"✅ Generated SQL:\n{sql_match.group(1).strip()}")
		else:
			print("⚠️ No SQL block found in response.")

		# Check if Plotly HTML component was generated
		if "```html" in response and "Plotly" in response:
			print("✅ Plotly HTML Chart component generated successfully.")
		else:
			print("ℹ️ No Plotly chart generated (or chart_type was 'none').")

		# Check if AST / Security blocked a query
		if "Security Exception" in response or "Prohibited" in response:
			print("🛡️ Security Guardrail correctly triggered.")

	except Exception as e:
		print(f"\n❌ UNHANDLED EXCEPTION DURING TEST: {e}")
	finally:
		asyncio.run(pipeline.on_shutdown())


if __name__ == "__main__":
	# Test 1: Standard read query (should succeed & generate chart/table)
	run_test("Which top 5 committees had the highest contributions in 2024?")

	# Test 2: Security injection test (should be blocked by sqlglot AST)
	# run_test("DROP TABLE aux_congress; SELECT * FROM fec_cm;")
