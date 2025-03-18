from functions import *
import sympy as sp
import re

x = sp.symbols('x0 x1')
l = sp.symbols('l')
v = [sp.symbols('v0'), sp.symbols('v1')]
z = [sp.symbols('z0'), sp.symbols('z1')]
w = sp.symbols('w')

def lagrange_function(x, l):
    """Функция Лагранжа"""
    return func1(x) + l * dop(x)

def compute_derivatives(x, l):
    """Вычисление частных производных из функции Лагранжа"""
    L_func = lagrange_function(x, l)
    
    dL_dx0 = sp.diff(L_func, x[0])
    dL_dx1 = sp.diff(L_func, x[1])
    dL_dl = sp.diff(L_func, l)
    return dL_dx0, dL_dx1, dL_dl


def modify_derivatives(dL_dx0, dL_dx1, dL_dl, v, z, w):
    """Модификация частных производных"""
    modified_dL_dx0 = dL_dx0 - v[0] + z[0]
    modified_dL_dx1 = dL_dx1 - v[1] + z[1]
    modified_dL_dl = dL_dl + w
    return modified_dL_dx0, modified_dL_dx1, modified_dL_dl

def modify_and_sum_derivatives(modified_dL_dx0, modified_dL_dx1):
    """Сложение частных производных с изменением знаков и удалением z"""
    modified_dL_dx0_with_sign = -modified_dL_dx0.subs({z[0]: 0, z[1]: 0})
    modified_dL_dx1_with_sign = -modified_dL_dx1.subs({z[0]: 0, z[1]: 0})
    
    result = modified_dL_dx0_with_sign + modified_dL_dx1_with_sign
    return result

def calculate_modified_sum_of_derivatives():
    """Целевая функция для вычисления всех шагов"""
    # Вычисление частных производных
    dL_dx0, dL_dx1, dL_dl = compute_derivatives(x, l)
    
    # Модификация частных производных
    modified_dL_dx0, modified_dL_dx1, modified_dL_dl0 = modify_derivatives(dL_dx0, dL_dx1, dL_dl, v, z, w)
    
    # Сложение частных производных с изменением знаков и удалением z
    result = modify_and_sum_derivatives(modified_dL_dx0, modified_dL_dx1)
    return result

def reorder_coefficients(expression):
    """Правильный порядок с изменением знаков, если нужно"""
    expression = expression.replace(' ', '')
    t=True
    terms = re.findall(r'[+-]?[\d]*\.?[\d]+\*?[a-zA-Z]+(?:\^?\d*)?|[+-]?[a-zA-Z]+\d*|[+-]?[\d]*\.?[\d]+', expression)

    constants = []
    x0_terms = []
    x1_terms = []
    l_terms = []
    v0_terms = []
    v1_terms = []
    w_terms = []
    z0_terms = []
    z1_terms = []
    for term in terms:
        if re.match(r'^-[\d]+(\.[\d]+)?$', term):
            t = False
            break

    for term in terms:
        if re.search(r'[a-zA-Z]', term):
            if t:
                if term.startswith('+'):
                    term = '-' + term[1:]
                elif term.startswith('-'):
                    term = '+' + term[1:]
                else:
                    term = '-' + term
            else:
                if term.startswith('+') or term.startswith('-'):
                    pass
                else:
                    term = '+' + term

            if 'x0' in term:
                x0_terms.append(term)
            elif 'x1' in term:
                x1_terms.append(term)
            elif 'l' in term:
                l_terms.append(term)
            elif 'v0' in term:
                v0_terms.append(term)
            elif 'v1' in term:
                v1_terms.append(term)
            elif 'w' in term:
                w_terms.append(term)
            elif 'z0' in term:
                z0_terms.append(term)
            elif 'z1' in term:
                z1_terms.append(term)
        else: 
            if term.startswith('-'):
                term = '+' + term[1:]
            constants.append(term)

    ordered_coeffs = [0] * 9
    ordered_coeffs[0] = constants[0] if constants else '+0'
    ordered_coeffs[1] = x0_terms[0] if x0_terms else '+0' 
    ordered_coeffs[2] = x1_terms[0] if x1_terms else '+0' 
    ordered_coeffs[3] = l_terms[0] if l_terms else '+0'
    ordered_coeffs[4] = v0_terms[0] if v0_terms else '+0' 
    ordered_coeffs[5] = v1_terms[0] if v1_terms else '+0' 
    ordered_coeffs[6] = w_terms[0] if w_terms else '+0'
    ordered_coeffs[7] = z0_terms[0] if z0_terms else '+0' 
    ordered_coeffs[8] = z1_terms[0] if z1_terms else '+0' 
    return ordered_coeffs


def extract_all_values(polynomial):
    """Нахождение коэффициентов в полиноме"""
    if isinstance(polynomial, list):
        polynomial = ''.join(polynomial)
    terms = re.findall(r'[+-]?[\d]*\.?[\d]*\*?[a-zA-Z]?\^?\d*', polynomial)
    all_values = []

    for term in terms:
        if not term:
            continue
        if re.search(r'[a-zA-Z]', term):
            coef_match = re.match(r'([+-]?[\d]*\.?[\d]*)', term)
            if coef_match:
                coef_value = coef_match.group(0)
                if coef_value == '' or coef_value == '+':
                    all_values.append(1.0)
                elif coef_value == '-':
                    all_values.append(-1.0)
                else:
                    all_values.append(float(coef_value))
        else:
            try:
                all_values.append(float(term))
            except ValueError:
                print(f"Warning: unable to convert '{term}' to float.")
    return all_values

def vect():
    """Извлечение коэффициентов из целевой функции"""
    str_f = str(calculate_modified_sum_of_derivatives())
    order = reorder_coefficients(str_f)
    coeff = extract_all_values(order)
    return coeff

def extract_and_modify2():
    """Извлечение коэффициентов из вспомогательной задачи ЛП"""
    dL_dx0, dL_dx1, dL_dl = compute_derivatives(x, l)
    modified_dL_dx0, modified_dL_dx1, modified_dL_dl = modify_derivatives(dL_dx0, dL_dx1, dL_dl, v, z, w)

    str_x0 = str(modified_dL_dx0)
    str_x1= str(modified_dL_dx1)
    str_l = str(modified_dL_dl)

    order1 = reorder_coefficients(str_x0)
    order2 = reorder_coefficients(str_x1)
    order3 = reorder_coefficients(str_l)
    coeff1= extract_all_values(order1)
    coeff2= extract_all_values(order2)
    coeff3= extract_all_values(order3)
    matrix = [coeff1, coeff2, coeff3]
    return matrix
