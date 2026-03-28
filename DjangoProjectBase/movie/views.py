from django.shortcuts import render
from django.http import HttpResponse

from .models import Movie

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64

from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

def home(request):
    #return HttpResponse('<h1>Welcome to Home Page</h1>')
    #return render(request, 'home.html')
    #return render(request, 'home.html', {'name':'Paola Vallejo'})
    searchTerm = request.GET.get('searchMovie') # GET se usa para solicitar recursos de un servidor
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm':searchTerm, 'movies':movies})


def about(request):
    #return HttpResponse('<h1>Welcome to About Page</h1>')
    return render(request, 'about.html')

def recommendations(request):
    #return HttpResponse('<h1>Welcome to Recommendations Page</h1>')
    return render(request, 'recommendations.html')

def signup(request):
    email = request.GET.get('email') 
    return render(request, 'signup.html', {'email':email})


def statistics_view0(request):
    matplotlib.use('Agg')
    # Obtener todas las películas
    all_movies = Movie.objects.all()

    # Crear un diccionario para almacenar la cantidad de películas por año
    movie_counts_by_year = {}

    # Filtrar las películas por año y contar la cantidad de películas por año
    for movie in all_movies:
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    # Ancho de las barras
    bar_width = 0.5
    # Posiciones de las barras
    bar_positions = range(len(movie_counts_by_year))

    # Crear la gráfica de barras
    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')

    # Personalizar la gráfica
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)

    # Ajustar el espaciado entre las barras
    plt.subplots_adjust(bottom=0.3)

    # Guardar la gráfica en un objeto BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convertir la gráfica a base64
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    # Renderizar la plantilla statistics.html con la gráfica
    return render(request, 'statistics.html', {'graphic': graphic})

def statistics_view(request):
    matplotlib.use('Agg')
    # Gráfica de películas por año
    all_movies = Movie.objects.all()
    movie_counts_by_year = {}
    for movie in all_movies:
        print(movie.genre)
        year = movie.year if movie.year else "None"
        if year in movie_counts_by_year:
            movie_counts_by_year[year] += 1
        else:
            movie_counts_by_year[year] = 1

    year_graphic = generate_bar_chart(movie_counts_by_year, 'Year', 'Number of movies')

    # Gráfica de películas por género
    movie_counts_by_genre = {}
    for movie in all_movies:
        # Obtener el primer género
        genres = movie.genre.split(',')[0].strip() if movie.genre else "None"
        if genres in movie_counts_by_genre:
            movie_counts_by_genre[genres] += 1
        else:
            movie_counts_by_genre[genres] = 1

    genre_graphic = generate_bar_chart(movie_counts_by_genre, 'Genre', 'Number of movies')

    return render(request, 'statistics.html', {'year_graphic': year_graphic, 'genre_graphic': genre_graphic})


def generate_bar_chart(data, xlabel, ylabel):
    keys = [str(key) for key in data.keys()]
    plt.bar(keys, data.values())
    plt.title('Movies Distribution')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

# Cargar la API Key
load_dotenv('../openAI.env')
client = OpenAI(api_key=os.environ.get('openai_apikey'))

# Función para calcular similitud de coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# views.py - Versión mejorada de recommend_movie

def recommend_movie(request):
    """Vista que maneja recomendaciones por título existente o descripción libre"""
    best_movie = None
    max_similarity = -1
    search_term = ''
    matched_by_title = False
    recommendations = []  # Para almacenar múltiples recomendaciones

    if request.method == 'POST':
        # Recibir el prompt del usuario
        search_term = request.POST.get('prompt', '').strip()
        
        if search_term:
            # PRIMERO: Verificar si el prompt coincide con algún título exacto o parcial
            # Intentamos encontrar coincidencia exacta primero
            exact_match = Movie.objects.filter(title__iexact=search_term).first()
            
            if exact_match:
                # Si hay coincidencia exacta con el título, usamos el embedding de esa película
                matched_by_title = True
                if exact_match.emb:
                    prompt_emb = np.frombuffer(exact_match.emb, dtype=np.float32)
                    
                    # Recorrer todas las películas excepto la que estamos usando como referencia
                    for movie in Movie.objects.exclude(id=exact_match.id):
                        if movie.emb:
                            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                            similarity = cosine_similarity(prompt_emb, movie_emb)
                            recommendations.append({
                                'movie': movie,
                                'similarity': similarity
                            })
                    
                    # Ordenar recomendaciones por similitud (mayor a menor)
                    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    # Tomar las 5 mejores recomendaciones
                    recommendations = recommendations[:5]
                    
                    # Si hay al menos una recomendación, la mejor es la primera
                    if recommendations:
                        best_movie = recommendations[0]['movie']
                        max_similarity = recommendations[0]['similarity']
            else:
                # Si no hay coincidencia con título, buscar coincidencia parcial
                title_matches = Movie.objects.filter(title__icontains=search_term)
                
                if title_matches.exists():
                    # Si hay múltiples coincidencias parciales, mostrar todas como opciones
                    matched_by_title = True
                    title_movies = list(title_matches)
                    
                    # Para cada coincidencia, generar recomendaciones basadas en su embedding
                    for title_match in title_movies:
                        if title_match.emb:
                            prompt_emb = np.frombuffer(title_match.emb, dtype=np.float32)
                            
                            for movie in Movie.objects.exclude(id=title_match.id):
                                if movie.emb:
                                    movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                                    similarity = cosine_similarity(prompt_emb, movie_emb)
                                    recommendations.append({
                                        'movie': movie,
                                        'similarity': similarity,
                                        'based_on': title_match.title
                                    })
                    
                    # Eliminar duplicados (si una misma película aparece por diferentes títulos)
                    unique_recommendations = {}
                    for rec in recommendations:
                        movie_id = rec['movie'].id
                        if movie_id not in unique_recommendations or rec['similarity'] > unique_recommendations[movie_id]['similarity']:
                            unique_recommendations[movie_id] = rec
                    
                    recommendations = list(unique_recommendations.values())
                    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
                    recommendations = recommendations[:5]
                    
                    if recommendations:
                        best_movie = recommendations[0]['movie']
                        max_similarity = recommendations[0]['similarity']
                
                else:
                    # No hay coincidencia con título, procesar como descripción libre
                    matched_by_title = False
                    
                    # Generar embedding del prompt del usuario
                    try:
                        response = client.embeddings.create(
                            input=[search_term],
                            model="text-embedding-3-small"
                        )
                        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)
                        
                        # Recorrer la base de datos y comparar con todas las películas
                        for movie in Movie.objects.all():
                            if movie.emb:
                                movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                                similarity = cosine_similarity(prompt_emb, movie_emb)
                                recommendations.append({
                                    'movie': movie,
                                    'similarity': similarity
                                })
                        
                        # Ordenar recomendaciones por similitud
                        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
                        
                        # Tomar las 5 mejores recomendaciones
                        recommendations = recommendations[:5]
                        
                        if recommendations:
                            best_movie = recommendations[0]['movie']
                            max_similarity = recommendations[0]['similarity']
                    
                    except Exception as e:
                        print(f"Error al generar embedding: {e}")
                        # Manejar error en caso de que falle la API
                        pass

    return render(request, 'recommendations.html', {
        'best_movie': best_movie,
        'similarity': max_similarity,
        'search_term': search_term,
        'matched_by_title': matched_by_title,
        'recommendations': recommendations,
        'has_recommendations': len(recommendations) > 0
    })