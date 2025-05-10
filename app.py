from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import ast
import random

app = Flask(__name__)

# Load your data
DATA_PATH = R"meal3_mealTime.csv"
df = pd.read_csv(DATA_PATH)

def filter_meals_by_diet(diet_types, meal_time):
    # Ensure 'meal_time_categories' is a list
    def parse_list(value):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)  # Safely convert string to list
            except (ValueError, SyntaxError):
                return []  # Return an empty list if parsing fails
        return value

    df['meal_time_categories'] = df['meal_time_categories'].apply(parse_list)
    
    # Filter meals by diet types and meal time
    filtered_meals = df[(df['diet_type'].isin(diet_types)) & 
                        (df['meal_time_categories'].apply(lambda x: meal_time in x))]
    return filtered_meals


class DietEnvironment:
    def __init__(self, meals, total_calories, total_budget, meal_times):
        self.meals = meals
        self.total_calories = total_calories
        self.total_budget = total_budget
        self.meal_times = meal_times  # List of meal times (e.g., ['breakfast', 'lunch', 'dinner'])
        self.current_calories = {meal_time: 0 for meal_time in meal_times}
        self.current_budget = 0
        self.selected_meals = {meal_time: [] for meal_time in meal_times}
        self.current_meal_time = 0  # Index to track the current meal time

    def reset(self):
        self.current_calories = {meal_time: 0 for meal_time in self.meal_times}
        self.current_budget = 0
        self.selected_meals = {meal_time: [] for meal_time in self.meal_times}
        self.current_meal_time = 0
        return self.get_state()

    def get_state(self):
        if self.current_meal_time >= len(self.meal_times):
            return (0, self.current_budget)  # Return a default state when out of bounds
        meal_time = self.meal_times[self.current_meal_time]
        return (self.current_calories[meal_time], self.current_budget)

    def step(self, action):
        if self.current_meal_time >= len(self.meal_times):
            return self.get_state(), 0, True  # Return a default state if out of bounds

        meal_time = self.meal_times[self.current_meal_time]
        meal = self.meals.iloc[action]
        calories = meal['calories_per_serving']
        cost = meal['cost_per_serving_in_inr']

        # Add meal to the plan
        self.selected_meals[meal_time].append(meal['meal_name'])
        self.current_calories[meal_time] += calories
        self.current_budget += cost

        # Calculate reward
        if self.current_budget > self.total_budget:
            reward = -20  # Penalty for exceeding budget
            done = True
        elif self.current_calories[meal_time] >= 0.8 * self.total_calories:  # Flexible calorie goal
            reward = 50  # Reward for meeting calorie goal
            self.current_meal_time += 1  # Move to the next meal time
            done = self.current_meal_time >= len(self.meal_times)  # Done if all meal times are covered
        else:
            reward = -10  # Small penalty for not meeting the goal yet
            done = False

        return self.get_state(), reward, done
    
class DietAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.q_table = {}  # Use a regular dictionary instead of defaultdict
        self.epsilon = 1.0  # Start with high exploration
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor

    def choose_action(self, state):
        state = str(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.n_actions)  # Initialize state if not present
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        state = str(state)
        next_state = str(next_state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)  # Initialize state if not present
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)  # Initialize next_state if not present
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(diet_type, total_calories, total_budget, meal_times, episodes=5000):
    meals_by_time = {meal_time: filter_meals_by_diet(diet_type, meal_time) for meal_time in meal_times}
    if any(meals.empty for meals in meals_by_time.values()):
        raise ValueError(f"No meals found for one or more meal times: {meal_times}")

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
    # Filter meals for each meal_time
    meals_by_time = {meal_time: filter_meals_by_diet(diet_types, meal_time) for meal_time in meal_times}
    
    # Check if any meal_time has no available meals
    for meal_time, meals in meals_by_time.items():
        if meals.empty:
            raise ValueError(f"No meals available for {meal_time}. Please check the dataset or constraints.")
    
    # Initialize a dictionary to store the selected meals
    selected_meals_details = {meal_time: [] for meal_time in meal_times}
    total_selected_calories = 0
    used_meals = set()  # To track meals that have already been selected

    # Step 1: Ensure at least one meal for each meal_time
    for meal_time in meal_times:
        available_meals = meals_by_time[meal_time]
        if available_meals.empty:
            continue

        # Exclude already used meals
        available_meals = available_meals[~available_meals['meal_name'].isin(used_meals)]

        # Shuffle the available meals to introduce randomness
        available_meals = available_meals.sample(frac=1).reset_index(drop=True)

        # Select the highest-calorie meal for the current meal_time
        if not available_meals.empty:
            meal = available_meals.iloc[0]
            selected_meals_details[meal_time].append({
                'meal_name': meal['meal_name'],
                'calories_per_serving': meal['calories_per_serving'],
                'cost_per_serving_in_inr': meal['cost_per_serving_in_inr'],
                'meal_time': meal_time
            })
            used_meals.add(meal['meal_name'])
            total_selected_calories += meal['calories_per_serving']

    # Step 2: Add a second meal to some meal_times if calories are not satisfied
    if total_selected_calories < total_calories:
        remaining_calories = total_calories - total_selected_calories
        for meal_time in meal_times:
            available_meals = meals_by_time[meal_time]
            available_meals = available_meals[~available_meals['meal_name'].isin(used_meals)]
            available_meals = available_meals.sample(frac=1).reset_index(drop=True)  # Shuffle meals

            for _, meal in available_meals.iterrows():
                if len(selected_meals_details[meal_time]) < 2 and remaining_calories > 0:
                    selected_meals_details[meal_time].append({
                        'meal_name': meal['meal_name'],
                        'calories_per_serving': meal['calories_per_serving'],
                        'cost_per_serving_in_inr': meal['cost_per_serving_in_inr'],
                        'meal_time': meal_time
                    })
                    used_meals.add(meal['meal_name'])
                    total_selected_calories += meal['calories_per_serving']
                    remaining_calories -= meal['calories_per_serving']

                # Stop if total calories are satisfied
                if total_selected_calories >= total_calories:
                    break
            if total_selected_calories >= total_calories:
                break

    # Flatten the selected meals into a single DataFrame
    diet_plan_df = pd.DataFrame([meal for meals in selected_meals_details.values() for meal in meals])
    return diet_plan_df

@app.route('/')
def home():
    return "Diet Recommendation API is running!"

@app.route('/recommend_diet', methods=['POST', 'GET'])
def get_recommendation():
    if request.method == 'GET':
        return "Please send a POST request with required parameters."
    data = request.json
    print("📥 Received JSON:", data)  # 🧾 Log incoming data

    diet_type = data['diet_type']
    total_calories = data['total_calories']
    total_budget = data['total_budget']
    meal_times = data['meal_times']
    try:
        agent = train_agent(diet_type, total_calories, total_budget, meal_times)

        diet_plan_details = generate_diet_plan(
            agent = agent,
            diet_types = diet_type, 
            total_calories = total_calories, 
            total_budget = total_budget, 
            meal_times = meal_times

        )
        diet_plan_df = diet_plan_details.reset_index(drop=True)
        return jsonify(diet_plan_df.to_dict(orient="records"))


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

