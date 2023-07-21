from nemoguardrails.actions import action

# Just a demo action that returns False    

@action()
async def is_admin():
    # Validate auth

    return False
