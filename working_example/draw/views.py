# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader



def index(request):
    latest_question_list = ["Hello", "Hi"]
    template = loader.get_template('draw/index.html')
    context = {
        'latest_question_list': latest_question_list,
    }
    return HttpResponse(template.render(context, request))

# Create your views here.
