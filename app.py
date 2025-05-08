from flask import Flask, request, jsonify
import pandas as pd
import ast
import random

app = Flask(__name__)

# Load your data
DATA_PATH = R"meal3_mealTime.csv"
df = pd.read_csv(DATA_PATH)

# Helper Functions
def personalize_calorie_goal(weight_kg, sex, goal):
    bmr = 22 * weight_kg  # Simplified BMR estimate
    if goal == 'gain':
        return bmr + 500
    elif goal == 'lose':
        return bmr - 500
    else:
        return bmr

def filter_meals_by_diet(df, is_veg=True, meal_time='lunch'):
    df = df.copy()
    df['meal_time_categories'] = df['meal_time_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df = df[df['diet_type'].str.lower() == ('vegetarian' if is_veg else 'non-vegetarian')]
    return df[df['meal_time_categories'].apply(lambda x: meal_time in x)].reset_index(drop=True)

# Environment Class
class DietEnvironment:
    def __init__(self, meals_df, calorie_goal, budget):
        self.meals_df = meals_df
        self.calorie_goal = calorie_goal
        self.budget = budget
        self.reset()

    def reset(self):
        self.total_calories = 0
        self.total_cost = 0
        self.selected_meals = []
        self.remaining_meals = self.meals_df.copy()
        return self._get_state()

    def _get_state(self):
        return [self.total_calories, self.total_cost]

    def step(self, action_idx):
        if action_idx >= len(self.remaining_meals):
            return self._get_state(), -10, True, {}

        meal = self.remaining_meals.iloc[action_idx]
        self.selected_meals.append(meal['meal_name'])
        self.total_calories += meal['calories_per_serving']
        self.total_cost += meal['cost_per_serving_in_inr']
        self.remaining_meals = self.remaining_meals.drop(self.remaining_meals.index[action_idx]).reset_index(drop=True)

        done = self.total_calories >= self.calorie_goal or self.total_cost >= self.budget
        reward = -abs(self.total_calories - self.calorie_goal) - max(0, self.total_cost - self.budget)

        return self._get_state(), reward, done, {}

# Recommendation Logic
def recommend_diet(df, weight_kg, sex, goal, is_veg, meal_time, budget):
    calorie_goal = personalize_calorie_goal(weight_kg, sex, goal)
    meals_df = filter_meals_by_diet(df, is_veg, meal_time)

    env = DietEnvironment(meals_df, calorie_goal, budget)
    state = env.reset()

    while True:
        if len(env.remaining_meals) == 0:
            break
        action = random.randint(0, len(env.remaining_meals) - 1)
        next_state, reward, done, _ = env.step(action)
        if done:
            break

    return env.selected_meals, reward, calorie_goal, env.total_calories, env.total_cost

# API Endpoint

@app.route('/')
def home():
    return "Diet Recommendation API is running!"

@app.route('/recommend_diet', methods=['POST', 'GET'])
def get_recommendation():
    data = request.json
    try:
        meals, reward, goal, final_cals, spent = recommend_diet(
            df=df,
            weight_kg=data['weight_kg'],
            sex=data['sex'],
            goal=data['goal'],
            is_veg=data['is_veg'],
            meal_time=data['meal_time'],
            budget=data['budget']
        )

        return jsonify({
            "recommended_meals": meals,
            "calorie_goal": goal,
            "calories_consumed": final_cals,
            "budget_spent": spent,
            "reward": reward
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
