# package imports
from dash import html, dcc, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from flask_login import UserMixin, current_user, logout_user, login_user

class User(UserMixin):
    # User data model
    def __init__(self, username):
        self.id = username
        self.role = 'user'  # Default role, can be extended

login_card = dbc.Card(
    [
        dbc.CardHeader("Login"),
        dbc.CardBody(
            [
                dbc.Input(
                    placeholder="Username",
                    type="text",
                    id="login-username", # Input field for username
                    className="mb-2" # Bootstrap class for margin bottom, 2 = size of the margin
                ),
                dbc.Input(
                    placeholder="Password",
                    type="password",
                    id="login-password", # Input field for password
                    className="mb-2" # Bootstrap class for margin bottom, 2 = size of the margin
                ),
                dbc.Button(
                    "Login",
                    id="login-button", # Button to trigger login
                    n_clicks = 0, # Initialize click count to 0
                    color="info", # Button color
                    type="submit", # Button type
                    className="float-end"  # Floats button to the right
                ),
                html.Div(children="", id="output-state") # Div to display output messages
            ]
        )
    ]
)

login_location = dcc.Location(id="url-login") # Location component to handle URL changes
login_info = html.Div(id="user-status-header")
logged_in_info = html.Div(
    [
        dbc.Button(
            html.I(className="fa-solid fa-circle-user fa-xl"), # Font Awesome icon for user
            id="user-popover",
            outline=True,
            color="light", # Button color
            className="border-0", # No border
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("Settings"),
                dbc.PopoverBody(
                    [
                        dcc.Link(
                            [
                                html.I(className="fa-solid fa-arrow-right-from-bracket me-1"), # Font Awesome icon for logout
                                "Logout"
                            ],
                            href="/logout"
                        )
                    ]
                )
            ],
            target="user-popover", # Target the button for popover
            trigger="focus", # Trigger popover on focus
            placement="bottom", # Position the popover below the button
        )
    ]
)

logged_out_info = dbc.NavItem(
    dbc.NavLink(
        "Login",
        href="/login"
    )
)

# Update user status header based on login state
@callback(
    Output("user-status-header", "children"),
    Input("url-login", "pathname")
)
def update_authentication_status(path):
    logged_in = current_user.is_authenticated
    # if the user is logged in and wants to log out
    if path == "/logout" and logged_in:
        logout_user() # Log out the user if they were logged in
        child = logged_out_info # Re display login link
    # If the user is logged in
    elif logged_in:
        child = logged_in_info
    else:
        child = logged_out_info
    return child

@callback(
    Output("output-state", "children"),
    Output("url-login", "pathname"),
    Input("login-button", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    State("_pages_location", "pathname"),
    prevent_initial_call=True
)
def login_button_click(n_clicks, username, password, pathname):
    if n_clicks > 0:
        if username == "admin" and password == "admin":
            login_user(User(username))
            return "Login successful", "/"
        return "Incorrect username or password", pathname
    raise PreventUpdate # Prevent update if no clicks have occurred
