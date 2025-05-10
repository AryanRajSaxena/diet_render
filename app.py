from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import ast
import random

app = Flask(__name__)

# Load your data
DATA_PATH = "meal3_mealTime.csv"
df = pd.read_csv(DATA_PATH)

def filter_meals_by_diet(diet_types, meal_time):
    def parse_list(value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
        return value

    df['meal_time_categories'] = df['meal_time_categories'].apply(parse_list)
    filtered_meals = df[(df['diet_type'].isin(diet_types)) & 
                        (df['meal_time_categories'].apply(lambda x: meal_time in x))]
    return filtered_meals

class DietEnvironment:
    def __init__(self, meals, total_calories, total_budget, meal_times):
        self.meals = meals.reset_index(drop=True)
        self.total_calories = total_calories
        self.total_budget = total_budget
        self.meal_times = meal_times
        self.current_calories = {meal_time: 0 for meal_time in meal_times}
        self.current_budget = 0
        self.selected_meals = {meal_time: [] for meal_time in meal_times}
        self.current_meal_time = 0

    def reset(self):
        self.current_calories = {meal_time: 0 for meal_time in self.meal_times}
        self.current_budget = 0
        self.selected_meals = {meal_time: [] for meal_time in self.meal_times}
        self.current_meal_time = 0
        return self.get_state()

    def get_state(self):
        if self.current_meal_time >= len(self.meal_times):
            return (0, self.current_budget)
        meal_time = self.meal_times[self.current_meal_time]
        return (self.current_calories[meal_time], self.current_budget)

    def step(self, action):
        if self.current_meal_time >= len(self.meal_times):
            return self.get_state(), 0, True

        if action >= len(self.meals):
            return self.get_state(), -20, True  # Invalid action fallback

        meal_time = self.meal_times[self.current_meal_time]
        meal = self.meals.iloc[action]
        calories = meal['calories_per_serving']
        cost = meal['cost_per_serving_in_inr']

        self.selected_meals[meal_time].append(meal['meal_name'])
        self.current_calories[meal_time] += calories
        self.current_budget += cost

        if self.current_budget > self.total_budget:
            return self.get_state(), -20, True
        elif self.current_calories[meal_time] >= 0.8 * self.total_calories:
            self.current_meal_time += 1
            return self.get_state(), 50, self.current_meal_time >= len(self.meal_times)
        else:
            return self.get_state(), -10, False

class DietAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.9

    def choose_action(self, state):
        state = str(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        state = str(state)
        next_state = str(next_state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)

        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(diet_type, total_calories, total_budget, meal_times, episodes=500):
    meals_by_time = {mt: filter_meals_by_diet(diet_type, mt) for mt in meal_times}
    if any(m.empty for m in meals_by_time.values()):
        raise ValueError("No meals found for one or more selected meal times")

    env = DietEnvironment(pd.concat(meals_by_time.values()), total_calories, total_budget, meal_times)
    agent = DietAgent(len(env.meals))

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

    return agent

def generate_diet_plan(agent, diet_types, total_calories, total_budget, meal_times):
    meals_by_time = {mt: filter_meals_by_diet(diet_types, mt) for mt in meal_times}
    for mt, meals in meals_by_time.items():
        if meals.empty:
            raise ValueError(f"No meals available for {mt}")

    selected = []
    used = set()
    total_cals = 0

    for mt in meal_times:
        available = meals_by_time[mt][~meals_by_time[mt]['meal_name'].isin(used)]
        available = available.sample(frac=1).reset_index(drop=True)
        for _, meal in available.iterrows():
            if len([m for m in selected if m['meal_time'] == mt]) < 2:
                selected.append({
                    'meal_name': meal['meal_name'],
                    'calories_per_serving': int(meal['calories_per_serving']),
                    'cost_per_serving_in_inr': int(meal['cost_per_serving_in_inr']),
                    'meal_time': mt
                })
                used.add(meal['meal_name'])
                total_cals += int(meal['calories_per_serving'])
            if total_cals >= total_calories:
                break
        if total_cals >= total_calories:
            break

    return pd.DataFrame(selected)

@app.route('/')
def home():
    return "Diet Recommendation API is running!"

@app.route('/recommend_diet', methods=['POST'])
def get_recommendation():
    try:
        data = request.get_json()
        print("📥 Received JSON:", data)

        diet_type = data.get("diet_type", [])
        total_calories = int(data.get("total_calories", 0))
        total_budget = int(data.get("total_budget", 0))
        meal_times = data.get("meal_times", [])

        if not diet_type or not meal_times or total_calories <= 0 or total_budget <= 0:
            return jsonify({"error": "Invalid input"}), 400

        agent = train_agent(diet_type, total_calories, total_budget, meal_times)
        plan_df = generate_diet_plan(agent, diet_type, total_calories, total_budget, meal_times)
        return jsonify(plan_df.to_dict(orient="records"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
