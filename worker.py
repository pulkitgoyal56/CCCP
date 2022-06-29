#!/usr/bin/env python
# coding: utf-8

import numpy as np

import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

from tqdm.notebook import tqdm
# import winsound


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
actions = ActionChains(driver)

driver.get("https://kyblab.tuebingen.mpg.de/experiment_ccs2022/banditexperiment/")

## Function to press one of the two buttons for the task
act = lambda x: ActionChains(driver).key_down('j' if x else 'f').pause(0.2).key_up('j' if x else 'f').perform() # driver.execute_script(f"returnpressed=1; timeInMs=Date.now()-timeInMs; myfunc({int(x)});")

## Function to start experiment by skipping all the initial crap
start = lambda: driver.execute_script("begintrial(); clickStart('page1', 'page8')")

## Function to print current total-score on the console
# score = lambda: driver.execute_script("console.log(totalscore)")

## Function to print passed text on the console (for debugging)
# log = lambda t: driver.execute_script(f"console.log({t})")

## Function to beep (on Windows)
# beep = lambda: winsound.PlaySound('sound.wav', winsound.SND_FILENAME)

## Function to clear the progress bars on the webpage
# clean = lambda: driver.execute_script("$('#progressBarLearn').hide(); $('#progressBarPredict').hide()")

## Function to get the current attributes in a design matrix
getX = lambda: np.array([[float(driver.find_element("id", f"featues{i+1}{j+1}").get_attribute('innerHTML')) for j in range(4)] for i in range(2)])

## Function to get the target values after action
getY = lambda: np.array([float(driver.find_element("id", f"outcome{i+1}").get_attribute('innerHTML')) for i in range(2)])

## Function to confirm the alert message
alert = lambda: driver.switch_to.alert.accept()


class RescorlaWagner():
    def __init__(self, num_inputs, sigma_y=0.1, learning_rate=0.2):
        self.num_inputs = num_inputs
        self.sigma_y = sigma_y
        self.learning_rate = learning_rate
        
        self.weights = np.zeros((num_inputs, 1))
        
    def predict(self, inputs):
        return self.weights.T @ inputs
        
    def learn(self, inputs, targets):
        self.weights += self.learning_rate * (targets - self.predict(inputs)) * inputs


class KalmanFilter():
    def __init__(self, num_inputs, sigma_y=0.1, sigma_w=1):
        self.num_inputs = num_inputs
        self.sigma_y = sigma_y
        self.sigma_w = sigma_w
        
        self.weights = np.zeros((num_inputs, 1))
        self.covariance = sigma_w * np.eye(num_inputs)
        
    def predict(self, inputs):
        mean = self.weights.T @ inputs
        std = np.sqrt(inputs.T @ self.covariance @ inputs + self.sigma_y ** 2)
        return mean, std
        
    def learn(self, inputs, targets):
        kalman_numerator = self.covariance @ inputs
        kalman_denominator = inputs.T @ self.covariance @ inputs + self.sigma_y ** 2
        kalman_gain = kalman_numerator / kalman_denominator
        self.weights = self.weights + kalman_gain * (targets - self.weights.T @ inputs)
        self.covariance = self.covariance - kalman_gain @ inputs.T @ self.covariance


if __name__ == '__main__':
    N_BLOCKS = 30
    N_TRIALS = 10

    start()
    for block in tqdm(range(N_BLOCKS)):
        model = KalmanFilter(4)
        for trial in tqdm(range(N_TRIALS), leave=False):
            time.sleep(0.1+np.random.rand()*0.3)

            X = getX()
            decision = (model.predict(X[[1]].T)[0] > model.predict(X[[0]].T)[0]).squeeze()
            if np.random.rand() > 0.9: decision = ~decision # Deliberate mistakes 
            act(decision); # beep()

            WebDriverWait(driver, 7).until(lambda driver: driver.find_element("id", f"outcome1").get_attribute('innerHTML') != '')
            Y = getY()
            model.learn(X[[0]].T, Y[0]), model.learn(X[[1]].T, Y[1])
            # score()

            if trial != N_TRIALS-1: WebDriverWait(driver, 7).until(lambda driver: driver.find_element("id", f"outcome1").get_attribute('innerHTML') == '')

        if block != N_BLOCKS-1:
            WebDriverWait(driver, 7).until(EC.alert_is_present())
            time.sleep(1)
            alert()
