import os
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = 'Update movie images from the local folder'

    def handle(self, *args, **options):
        images_folder = 'media/movie/images/'

        movies = Movie.objects.all()
        updated_count = 0

        for movie in movies:
            image_filename = f"m_{movie.title}.png"
            image_relative_path = os.path.join('movie/images', image_filename)
            image_full_path = os.path.join(images_folder, image_filename)

            if os.path.exists(image_full_path):
                movie.image = image_relative_path
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"Updated image for: {movie.title}"))
                updated_count += 1
            else:
                self.stdout.write(self.style.WARNING(f"Image not found for: {movie.title}"))

        self.stdout.write(self.style.SUCCESS(f"✅ Completed. Total movies updated: {updated_count}"))
