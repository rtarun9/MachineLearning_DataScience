import matplotlib.pyplot as plt
import numpy as np

# equally spaced data between 0 and 10
x = np.linspace(0, 10, 100)
print(x)

# sin(x) and cos(x)
y = np.sin(x)
z = np.cos(x)

# plot x vs sin(x) and x vs cos(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('x  vs sin(x)')
plt.show()

plt.plot(x, z)
plt.title('x vs cos(x)')
plt.xlabel('x')
plt.ylabel('cos(x)')
plt.show()

# bar plot
x = ['English', 'French', 'Spanish', 'Latin', 'German']
y = [20, 40, 20, 60, 80]
plt.bar(x, y)
plt.xlabel('languages')
plt.ylabel('population (in m)')
plt.title('language vs population')
plt.show()