from rlgym.utils.terminal_conditions.common_conditions \
    import NoTouchTimeoutCondition, GoalScoredCondition, TimeoutCondition


def AasmaTerminalCondition(tick_skip: int = 8):
    return (
        NoTouchTimeoutCondition(round(30 * 120 / tick_skip)),
        GoalScoredCondition()
    )


def AasmaHumanTerminalCondition(tick_skip: int = 8):
    return (
        TimeoutCondition(round(30 * 120 / tick_skip)),
        GoalScoredCondition()
    )
