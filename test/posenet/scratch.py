import math

point1 = (40.730424, -73.997221) # Lat/Long (lambda/phi)
point2 = (40.728553, -73.99454) # Lat/Long (lambda/phi)

r = 6371000 # meters
phi_0 = point1[1]
cos_phi_0 = math.cos(math.radians(phi_0))

def to_xy(point, r, cos_phi_0):
    lam = point[0]
    phi = point[1]
    return (r * math.radians(lam) * cos_phi_0, r * math.radians(phi))

point1_xy = to_xy(point1, r, cos_phi_0)
point2_xy = to_xy(point2, r, cos_phi_0)
#-57.354869 298.113598
dist = math.sqrt((-57.354869)**2 + (298.113598)**2)
print(dist)