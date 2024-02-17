from django.db import models

from datetime import datetime

def omr_upload_location(instance, filename, **kwargs):
    file_path = f"image/{datetime.now().timestamp()}-{filename}"
    return file_path

class ImageModel(models.Model):
    image = models.ImageField(upload_to=omr_upload_location)
    name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(
        auto_now_add=True, verbose_name="created_at"
    )
    updated_at = models.DateTimeField(auto_now=True, verbose_name="updated_at")
