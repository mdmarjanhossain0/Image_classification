from django.shortcuts import render

from omr.models import ImageModel
from omr.worker.worker import classify_image









def home_view(request):
    context = {}
    if request.method == "POST":
        print(request.POST)
        print(request.FILES)
        image_file = request.FILES.get("image")
        image = ImageModel.objects.create(image=image_file)
        print(image.image.path)
        answer = classify_image(image.image.path)
        print(answer)
        image.name = answer
        image.save()
        context["output_class"] = answer
        context["image_url"] = image.image.url

    return render(request, "home.html", context)


def controll_view(request):
    context = {}
    if request.method == "POST":
        print(request.POST)
        if request.POST.get("start", None):
            print("start")
            context["status"] = "started"
        else:
            print("stop")
            context["status"] = "stoped"

    return render(request, "controller.html", context)