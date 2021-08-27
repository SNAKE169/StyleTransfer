from django import forms
 
class StyleTransferForm(forms.Form):
    image = forms.ImageField()
    style = forms.ImageField()