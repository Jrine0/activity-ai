from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('maze/', views.maze_solver, name='maze_solver'),
    path('api/solve_maze/', views.solve_maze_api, name='solve_maze_api'),
    path('garbage/', views.garbage_truck, name='garbage_truck'),
    path('api/solve_garbage/', views.solve_garbage_api, name='solve_garbage_api'),
    path('api/generate_city/', views.generate_city_api, name='generate_city_api'),
    path('parking/', views.parking_finder, name='parking_finder'),
    path('api/solve_parking/', views.solve_parking_api, name='solve_parking_api'),
    path('api/generate_parking/', views.generate_parking_api, name='generate_parking_api'),
    path('diagnosis/', views.fault_diagnosis, name='fault_diagnosis'),
    path('api/solve_diagnosis/', views.solve_diagnosis_api, name='solve_diagnosis_api'),
    path('automation/', views.home_automation, name='home_automation'),
    path('api/solve_automation/', views.solve_automation_api, name='solve_automation_api'),
]
