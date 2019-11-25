from django.http import HttpResponse
from peoplePredict.model.null_model import NullModel
import json


def hello(request):
    model = NullModel()
    res = {'model': model.name,
           'param': model.predict(request.GET)}
    return HttpResponse(json.dumps(res))
