import numpy as np
from p5 import *


def load_image_rgb(path):
    img = load_image(path)
    img.pixels[:, :, [0, 1, 2]] = img.pixels[:, :, [2, 1, 0]]  # Convert from BGR to RGB
    return img


def setup():
    global car_img, charger_img
    global cars, chargers, distances, pheromones, shortest_path, shortest_distance
    global ants, num_ants, dst_power, pheromone_power, evaporation_rate, pheromone_intensity

    shortest_path = None
    shortest_distance = float('inf')
    cars, chargers, distances, pheromones = np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 0, 1)), np.empty((0, 0, 1))

    ants = []
    num_ants = 1000
    dst_power, pheromone_power, evaporation_rate, pheromone_intensity = 8.0, 4.0, 0.2, 2.0

    car_img = load_image_rgb("img/car.png")
    charger_img = load_image_rgb("img/charger.png")

    size(1280, 720)
    set_frame_rate(1000)


def spawn_ants():
    global num_ants, ants, chargers, cars, shortest_path, shortest_distance
    ants = []
    if len(chargers) > 0 and len(cars) > 0:
        charger_slots = {i: 2 for i in range(len(chargers))}
        for _ in range(num_ants):
            spawn_location = {'type': 'car', 'id': np.random.randint(0, len(cars))}
            ant = {
                'distanceTraveled': 0.0,
                'carsVisited': set(),
                'chargerSlots': charger_slots.copy(),
                'location': spawn_location,
                'nodesVisited': [spawn_location.copy()]
            }
            ants.append(ant)


def reset_shortest_path():
    global shortest_path, shortest_distance
    shortest_path = None
    shortest_distance = float('inf')


def save_shortest_path():
    global ants, shortest_path, shortest_distance
    for ant in ants:
        if ant['distanceTraveled'] < shortest_distance:
            shortest_path = ant['nodesVisited']
            shortest_distance = ant['distanceTraveled']


def calculate_distances():
    global distances, cars, chargers
    if len(chargers) > 0 and len(cars) > 0:
        distances = abs(np.expand_dims(cars, axis=1) - chargers).sum(axis=2)


def initialize_pheromones():
    global pheromones
    pheromones = np.ones((len(cars), len(chargers)))


def mouse_pressed():
    global mouse_button, mouse_x, mouse_y, cars, chargers
    if mouse_button == LEFT:
        cars = np.vstack([cars, [(mouse_x, mouse_y)]])
    elif mouse_button == RIGHT:
        chargers = np.vstack([chargers, [(mouse_x, mouse_y)]])

    reset_shortest_path()
    calculate_distances()
    initialize_pheromones()
    spawn_ants()


def calculate_probabilites(neighbor_distances, neighbor_pheromones):
    global dst_power, pheromone_power
    desirabilities = ((1 / neighbor_distances)**dst_power) * (neighbor_pheromones**pheromone_power)
    desirabilities[desirabilities == 0] = 1e-10  # Avoid division by zero
    return desirabilities / sum(desirabilities)  # Normalize to probabilities


def update_pheromones():
    global ants, pheromones, evaporation_rate, pheromone_intensity
    pheromones *= 1 - evaporation_rate
    for ant in ants:
        for idx in range(len(ant['nodesVisited']) - 1):
            curr = ant['nodesVisited'][idx]
            next = ant['nodesVisited'][idx + 1]
            if curr['type'] == 'car':
                carId, chargerId = curr['id'], next['id']
            else:
                carId, chargerId = next['id'], curr['id']
            pheromones[carId][chargerId] += pheromone_intensity / ant['distanceTraveled']


def step():
    global ants, pheromones, chargers, cars, distances
    for ant in ants:
        if len(ant['carsVisited']) != len(cars) and sum(ant['chargerSlots'].values()) > 0.0:
            if ant['location']['type'] == 'car':
                possibleChargers = [i for i in range(len(chargers)) if ant['chargerSlots'][i] > 0]
                chargerDistances = distances[ant['location']['id']][possibleChargers]
                chargerPheromones = pheromones[ant['location']['id']][possibleChargers]
                probabilites = calculate_probabilites(chargerDistances, chargerPheromones)
                nextCharger = np.random.choice(range(len(possibleChargers)), p=probabilites)
                nextChargerId = possibleChargers[nextCharger]

                ant['carsVisited'].add(ant['location']['id'])
                ant['chargerSlots'][nextChargerId] -= 1
                ant['location']['type'] = 'charger'
                ant['location']['id'] = nextChargerId
                ant['distanceTraveled'] += chargerDistances[nextCharger]

            elif ant['location']['type'] == 'charger':
                possibleCars = [i for i in range(len(cars)) if i not in ant['carsVisited']]
                carDistances = distances[:, ant['location']['id']][possibleCars]
                carPheromones = pheromones[:, ant['location']['id']][possibleCars]
                probabilites = calculate_probabilites(carDistances, carPheromones)
                nextCar = np.random.choice(range(len(possibleCars)), p=probabilites)

                ant['location']['type'] = 'car'
                ant['location']['id'] = possibleCars[nextCar]
                ant['distanceTraveled'] += carDistances[nextCar]

            ant['nodesVisited'].append(ant['location'].copy())

    generation_done = True
    for ant in ants:
        if len(ant['carsVisited']) != len(cars) and sum(ant['chargerSlots'].values()) > 0.0:
            generation_done = False
            break

    if ants and generation_done:
        update_pheromones()
        save_shortest_path()
        spawn_ants()


def draw():
    global cars, chargers, pheromones, distances, shortest_path
    background(34, 39, 46)

    for car_id, (car_x, car_y) in enumerate(cars):
        for charger_id, (charger_x, charger_y) in enumerate(chargers):
            strokeWeight(pheromones[car_id][charger_id])
            stroke(1, 4, 9)
            line(car_x, car_y, charger_x, charger_y)

    if shortest_path:
        for idx in range(len(shortest_path) - 1):
            curr, next = shortest_path[idx], shortest_path[idx + 1]
            if curr['type'] == 'car':
                car_x, car_y = cars[curr['id']]
                charger_x, charger_y = chargers[next['id']]
            else:
                car_x, car_y = cars[next['id']]
                charger_x, charger_y = chargers[curr['id']]
            strokeWeight(8)
            stroke(218, 54, 51)
            line(car_x, car_y, charger_x, charger_y)

    for car_x, car_y in cars:
        image(car_img, car_x - car_img.width() // 2, car_y - car_img.height() // 2)

    for charger_x, charger_y in chargers:
        image(charger_img, charger_x - charger_img.width() // 2, charger_y - charger_img.height() // 2)

    step()


if __name__ == '__main__':
    run(renderer='skia')
