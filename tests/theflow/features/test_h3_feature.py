from theflow.features import h3_feature


def test_h3_to_list():
    assert h3_feature.H3FeatureMixin.h3_to_list(0) == [0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    assert h3_feature.H3FeatureMixin.h3_to_list(576495936675512319) == [
        1,
        0,
        0,
        0,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ]
    assert h3_feature.H3FeatureMixin.h3_to_list(102576495936675512319) == [
        1,
        7,
        8,
        71,
        2,
        7,
        1,
        2,
        2,
        6,
        1,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ]
    assert h3_feature.H3FeatureMixin.h3_to_list(50102576495936675512319) == [
        2,
        0,
        14,
        102,
        7,
        0,
        3,
        5,
        0,
        5,
        5,
        0,
        5,
        7,
        7,
        7,
        7,
        7,
        7,
    ]
