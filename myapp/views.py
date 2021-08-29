from django.shortcuts import render
from django.http import HttpResponse
from .forms import StyleTransferForm
from .models import StyleTransferModel

from StyleTransferModels.fast_transfer import transfer

def index(request):
    context = {}
    if request.method == 'POST':
        form = StyleTransferForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data.get("image")
            style = form.cleaned_data.get("style")
            # print('Checking ....')
            obj = StyleTransferModel.objects.create(img=img, style=style)
            # print('Checking ....')
            obj.save()
            # print('Checking...')
            img_url = obj.img.url
            style_url = obj.style.url
            generated_url = str(transfer(img_url, style_url))[5:]
            # print(img_url)
            # print(style_url)
            # print(img_url)
            # print(generated_url)
            return render(request, 'myapp/index.html', {'form': form, 'img_url': img_url, 'style_url': style_url, 'generated_url': generated_url})
    else:
        form = StyleTransferForm()
    context['form'] = form
    return render(request, 'myapp/index.html', context)

