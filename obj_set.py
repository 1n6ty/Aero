import bpy
import cmath

obj_name = "Sphere"
depth = 5
t = 0.001
k = 2
epsilon = 0.0000001

obj = None
for i in bpy.data.objects:
    if i.name == obj_name:
        obj = i
        break
if not obj:
    raise ValueError('No object found')

delta = (obj.bound_box[2][1] - obj.bound_box[0][1]) / depth
margin = delta / 2

response = [] # array with shape (1, depth * (k * 4 + 2)); each subarray with (k * 4 + 2) looks like [c_real (k * 2 + 1 times), c_imaginary (k * 2 + 1 times)]

width = (obj.bound_box[4][0] - obj.bound_box[0][0])
height = (obj.bound_box[1][2] - obj.bound_box[0][2])
for n in range(depth):
    origin = [0, obj.bound_box[0][1] + margin + n * delta, 0]
    response_buff = []
    for kb in range(-k, k + 1):
        if kb == 0:
            response_buff = [*response_buff[:len(response_buff) // 2], 0.5, *response_buff[len(response_buff) // 2:], 0.5]
            continue
        sum_N = 0j
        hit_count = 0
        for p in range(int(1 // t)):
            dir = cmath.exp(-2 * cmath.pi * t * p * 1j)
            dir = [dir.real, 0, dir.imag]
            origin_buff = origin[:]
            
            ray_buff = obj.ray_cast(origin_buff, dir)
            while ray_buff[0]:
                hit_count += 1
                
                sum_N += (ray_buff[1][0] + ray_buff[1][2] * 1j) * cmath.exp(kb * 2 * cmath.pi * t * p * 1j)
                
                origin_buff = [origin[0] + ray_buff[1][0] + dir[0] * epsilon, origin[1], origin[2] + ray_buff[1][2] + dir[2] * epsilon]
                ray_buff = obj.ray_cast(origin_buff, dir)
        response_buff = [*response_buff[:len(response_buff) // 2], sum_N.real / (hit_count * width), *response_buff[len(response_buff) // 2:], sum_N.imag / (hit_count * height)]
    response = [*response, *response_buff]
print(response)