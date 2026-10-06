''' Emotion_detector function '''

import requests
import json

def emotion_detector(text_to_analyze):
    # Primero cogemos el texto a analizar y lo preparamos en un paquete con las especificaciones requeridas.
    # El paquete se compone de una carga o payload.
    payload = {
              "raw_document": { "text": text_to_analyze }
    }

    # Y de unas variables para enviar al predictor de Watson
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    # Envia el mensaje y recibe la respuesta
    response = requests.post(
                             url,
                             headers=headers,
                             json=payload
    )
    # Obtiene la respuesta y la carga en un diccionario.
    response_dict = json.loads(response.text)

    # Obtenemos los scores de cada emocion.
    anger_score = response_dict["emotionPredictions"][0]["emotion"]["anger"]
    disgust_score = response_dict["emotionPredictions"][0]["emotion"]["disgust"]
    fear_score = response_dict["emotionPredictions"][0]["emotion"]["fear"]
    joy_score = response_dict["emotionPredictions"][0]["emotion"]["joy"]
    sadness_score = response_dict["emotionPredictions"][0]["emotion"]["sadness"]

    # Creamos una lista con los pares.
    emotions_list = {
                    'anger': anger_score,
                    'disgust': disgust_score,
                    'fear': fear_score,
                    'joy': joy_score,
                    'sadness': sadness_score,
    }

    # Buscamos la emocion dominante en la lista
    dominant_emotion = max(emotions_list, key=emotions_list.get)

    # Añadimos a la salida solicitada
    emotions_list["dominant_emotion"] = dominant_emotion

    return emotions_list
