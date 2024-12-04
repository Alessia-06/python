def neurone(input_v, weight, bias):
    return input_v * weight + bias

a = int(input('inserire input: '))
b = float(input('inserire peso: '))
c = int(input('inserire bias: '))

print(neurone(a, b, c))