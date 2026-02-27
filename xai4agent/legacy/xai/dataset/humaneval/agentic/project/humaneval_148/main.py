def bf(planet1, planet2):
    '''
    There are eight planets in our solar system: the closerst to the Sun
    is Mercury, the next one is Venus, then Earth, Mars, Jupiter, Saturn,
    Uranus, Neptune.
    Write a function that takes two planet names as strings planet1 and planet2.
    The function should return a tuple containing all planets whose orbits are
    located between the orbit of planet1 and the orbit of planet2, sorted by
    the proximity to the sun.
    The function should return an empty tuple if planet1 or planet2
    are not correct planet names.
    Examples
    bf("Jupiter", "Neptune") ==> ("Saturn", "Uranus")
    bf("Earth", "Mercury") ==> ("Venus")
    bf("Mercury", "Uranus") ==> ("Venus", "Earth", "Mars", "Jupiter", "Saturn")
    '''
    # 定义行星顺序
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

    # 检查输入是否为有效的行星名称
    if planet1 not in planets or planet2 not in planets:
        return ()

    # 获取行星在序列中的索引
    index1 = planets.index(planet1)
    index2 = planets.index(planet2)

    # 如果两个行星相同或相邻，则返回空元组
    if abs(index1 - index2) <= 1:
        return ()

    # 确保index1小于index2以便提取中间的行星
    start, end = sorted([index1, index2])

    # 返回位于两个行星之间的所有行星
    return tuple(planets[start+1:end])
