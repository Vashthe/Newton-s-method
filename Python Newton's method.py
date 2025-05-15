import sympy

# 牛顿迭代法
# 用户输入部分
func = input("请输入迭代函数：")  # 例如输入 "x**2 - 2" 或 "exp(x) - 2"
x_init = float(input("请输入初始值x0："))  # 例如输入 1.0
epsilon = float(input("请输入截断误差ε："))  # 例如输入 1e-6
N = int(input("请输入迭代次数N："))  # 例如输入 100

# 牛顿迭代法实现
x_value_1 = x_init  # 初始化迭代值
k = 1  # 迭代计数器
flag = False  # 收敛标志
x = sympy.symbols("x")  # 定义符号变量x

# 预计算函数表达式及其导数
exp_1 = sympy.sympify(func)  # 将输入字符串转换为SymPy表达式
exp_1_d1 = sympy.diff(exp_1, x)  # 计算一阶导数f'(x)

# 检查初始点导数是否为0
if abs(exp_1_d1.subs(x, x_init).evalf()) < 1e-12:
    print("牛顿下山法不收敛：初始点导数为零")
    exit()

while k < N:
    # 每次迭代重新计算函数值和导数值（可优化）
    fx = exp_1.subs(x, x_value_1).evalf()  # 计算f(x_k)
    fx_d1 = exp_1_d1.subs(x, x_value_1).evalf()  # 计算f'(x_k)

    # 牛顿迭代公式：x_{k+1} = x_k - f(x_k)/f'(x_k)
    x_pre = x_value_1  # 保存前一次迭代值
    x_value_1 = x_value_1 - fx / fx_d1  # 计算新迭代值

    # 收敛判断：相邻两次迭代值差小于阈值
    if abs(x_value_1 - x_pre) < epsilon:
        print(f"牛顿迭代法近似解的值为{x_value_1:.8f}，迭代了{k}次")
        flag = True
        break
    else:
        k += 1

if not flag:
    print(f"牛顿迭代法迭代失败：未在{N}次内收敛")

print("-" * 50)  # 分隔线

# 牛顿下山法
x_value_2 = x_init  # 初始化迭代值
k = 1  # 迭代计数器
flag = False  # 收敛标志
x = sympy.symbols("x")  # 定义符号变量x

# 预计算函数表达式及其导数
expr_2 = sympy.sympify(func)  # 原函数
expr_2_d1 = sympy.diff(expr_2, x)  # 一阶导数

# 检查初始点导数是否为0
if abs(expr_2_d1.subs(x, x_init).evalf()) < 1e-12:
    print("牛顿下山法不收敛：初始点导数为零")
    exit()


# 下山因子生成器（动态调整λ）
def lambda_generator():
    """生成递减的下山因子序列：1, 1/2, 1/4, 1/8,..."""
    lambda_val = 1.0
    while True:
        yield lambda_val
        lambda_val /= 2


lambda_gen = lambda_generator()

while k < N:
    fx_2 = expr_2.subs(x, x_value_2).evalf()  # 当前函数值
    fx_2_d1 = expr_2_d1.subs(x, x_value_2).evalf()  # 当前导数值
    x_pre = x_value_2

    # 尝试不同的下山因子直到满足下降条件
    for lambda_val in lambda_gen:
        x_value_2_new = x_value_2 - lambda_val * fx_2 / fx_2_d1  # 带下山因子的牛顿迭代
        fx_2_new = expr_2.subs(x, x_value_2_new).evalf()

        # 下山条件：保证|f(x_new)| < |f(x_value)|
        if abs(fx_2_new) < abs(fx_2):
            x_pre = x_value_2
            x_value_2 = x_value_2_new
            break

        # 如果lambda已经很小仍不满足条件，则退出
        if lambda_val < 1e-10:
            print("无法找到合适的下山因子")
            flag = True
            break

    # 检查是否收敛
    if flag or abs(x_value_2 - x_pre) < epsilon:
        print(f"牛顿下山法近似解的值为{x_value_2:.8f}，迭代了{k}次")
        flag = True
        break

    k += 1

if not flag:
    print(f"牛顿下山法迭代失败：未在{N}次内收敛")

print("-" * 50)