from django.urls import path
from . import views
from django.conf import settings          
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.Signup, name='signup'),
    path('login/', views.Login, name='login'),

    path('upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('preprocess/', views.preprocess_dataset, name='preprocess_dataset'),
    path('train-models/', views.train_models, name='train_models'),
    path("predict/", views.user_predict, name="user_predict"),

    path('admin-login/', views.admin_login, name='admin_login'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-logout/', views.admin_logout, name='admin_logout'),








]


  
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)