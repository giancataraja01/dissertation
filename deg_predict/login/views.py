from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from .forms import LoginForm
from upload_dataset.models import UploadedStudent

# Create your views here.


def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            uname = form.cleaned_data['username']
            pwd = form.cleaned_data['password']

            try:
                user = UploadedStudent.objects.get(
                username=uname, password=pwd)
                request.session['user_id'] = user.user_id
                request.session['username'] = user.username
                request.session['firstname'] = user.firstname
                request.session['lastname'] = user.lastname
                request.session['role'] = user.role
                request.session['coll'] = user.coll
                request.session['course']= user.course                

              #  messages.success(request, f"✅ Welcome {user.firstname}!")
                return redirect('dashboard')  # or your desired route
               # return render(request, "dashboard/dashboard.html", {'form': form})

            except UploadedStudent.DoesNotExist:
                messages.error(request, "❌ Invalid username or password.")
    else:
        form = LoginForm()

    return render(request, "login/login.html", {'form': form})

def user_logout(request):
    
    request.session.flush()
    logout(request)
    return redirect('login')  