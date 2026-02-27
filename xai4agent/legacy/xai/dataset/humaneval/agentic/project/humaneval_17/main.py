from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    # 定义音符到节拍数的映射
    note_to_beats = {
        'o': 4,   # whole note
        'o|': 2,  # half note
        '.|': 1   # quarter note
    }

    # 如果输入为空，返回空列表
    if not music_string:
        return []

    # 按空格分割字符串并转换为节拍数
    notes = music_string.split()
    return [note_to_beats[note] for note in notes]
