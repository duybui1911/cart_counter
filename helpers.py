import os
import pathlib
import json
import numpy as np
import openpyxl
import pandas as pd

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_coordinates_from_json(file_path):
    coordinates_list = []
    json_data = read_json(file_path)

    annotations = json_data.get('annotations', [])

    for annotation in annotations:
        segmentation = annotation.get('segmentation', [])
        coordinates_list.append(segmentation)
    
    coordinates_list = coordinates_list[0][0]
    coordinates = []
    for i in range(0, len(coordinates_list), 2):
        coordinates.append([coordinates_list[i], coordinates_list[i+1]])
    return np.array(coordinates, dtype=np.int16)

def convert_data_raw(results):
    str = ""
    #for key in config.keys():
    #val = results[key]
    str = f'{str}, {results["count"]}'
    return str

def write_title(path):
    #title_first = ","
    title_second = "No."
    #    title_first = f'{title_first}, {config[key]["name"]}, '
    title_second = f'{title_second}, Time, Quantity'
    with open(path, 'a', encoding='utf8') as file:
    #    file.write(title_first + "\n")
        file.write(title_second + "\n")

def convert_time(t_detect):
    
    h = t_detect//3600
    t_detect = t_detect -h
    p = t_detect//60
    s = t_detect%60
    if s<10:
        s=f'0{s}'
    if p<10:
        p=f'0{p}'
    if h<10:
        h=f'0{h}'
    str = f'{h}:{p}:{s}'
    return str
