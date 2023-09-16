import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

# Shader Compilation and Program Linking functions
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))

    return shader

def create_program(vertex_shader, fragment_shader):
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))

    return program

# Initialization
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Create two textures for ping-pong rendering
textures = glGenTextures(2)
for tex in textures:
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, display[0], display[1], 0, GL_RGB, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)

fbo = glGenFramebuffers(1)
# glBindFramebuffer(GL_FRAMEBUFFER, fbo)
# 
# glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[0], 0)
# if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
#     raise Exception("Error setting up framebuffer")

glBindFramebuffer(GL_FRAMEBUFFER, 0)

# Vertex Shader (used by both update and visualization)
vertex_shader_source = """
#version 300 es
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Update Shader (Fragment)
update_fragment_shader_source = """
#version 300 es
precision mediump float;

uniform float u_time;         
uniform vec2 u_resolution;   
uniform sampler2D u_prevState;

out vec4 fragColor;

vec2 reflectVec(vec2 dir, vec2 normal) {
    return dir - 2.0 * dot(dir, normal) * normal;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    vec2 obstacleCenter = vec2(0.7, 0.5);
    float obstacleRadius = 0.05;

    vec4 prevState = texture(u_prevState, uv);
    
    float currentDistance = prevState.r + u_time;
    float waveDirection = prevState.g;
    vec2 waveVector = vec2(cos(waveDirection), sin(waveDirection));

    vec2 currentPos = uv + waveVector * currentDistance;
    float distanceFromObstacle = distance(currentPos, obstacleCenter);

    if (distanceFromObstacle < obstacleRadius) {
        vec2 normal = normalize(currentPos - obstacleCenter);
        waveVector = reflectVec(waveVector, normal);
        waveDirection = atan(waveVector.y, waveVector.x);
    }
    
    fragColor = vec4(currentDistance, waveDirection, 0.0, 1.0);

    if (prevState.r == 0.0)
    {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // TODO
    fragColor = vec4(0.5, waveDirection, 0.0, 1.0);
}
"""

# Visualization Shader (Fragment)
visualization_fragment_shader_source = """
#version 300 es
precision mediump float;

uniform sampler2D u_state;   
uniform vec2 u_resolution;

out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    vec4 state = texture(u_state, uv);
    float distanceTravelled = state.r;

    // TODO: this 0.5 is just bullshit where we think we hit the obstacle
    if (abs(distanceTravelled - 0.5) < 0.01) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
"""

# Compile and link shaders
vs = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
fs = compile_shader(update_fragment_shader_source, GL_FRAGMENT_SHADER)
update_program = create_program(vs, fs)

vs_visualize = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
fs_visualize = compile_shader(visualization_fragment_shader_source, GL_FRAGMENT_SHADER)

visualize_program = create_program(vs_visualize, fs_visualize)

red_values = []

for y in range(display[1]):
    row = []
    for x in range(display[0]):
        if 300 <= x <= 350 and 300 <= y <= 350:
            row.append([0.5, 0.0, 0.0, 1.0])
        else:
            row.append([0.0, 0.0, 0.0, 1.0])
    red_values.append(row)

# Convert texture data to a numpy array
np_texture_data = np.array(red_values, dtype=np.float32)

# print(np_texture_data)

glBindTexture(GL_TEXTURE_2D, textures[1])
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, display[0], display[1], 0, GL_RGBA, GL_FLOAT, np_texture_data)
glBindTexture(GL_TEXTURE_2D, 0)

glBindTexture(GL_TEXTURE_2D, textures[0])
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, display[0], display[1], 0, GL_RGBA, GL_FLOAT, np_texture_data)
glBindTexture(GL_TEXTURE_2D, 0)

# retrieved_data = np.empty((display[1], display[0], 4), dtype=np.float32)
# glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, retrieved_data)
# 
# print(retrieved_data)
# 
# # Unbind the texture


# Main loop
current_texture = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
    if True:
        # 1. Update wave information
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)   # We're writing to a texture, not to the screen directly
        # glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[0], 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[1-current_texture], 0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(update_program)
        time_location = glGetUniformLocation(update_program, "u_time")
        resolution_location = glGetUniformLocation(update_program, "u_resolution")
        prev_state_location = glGetUniformLocation(update_program, "u_prevState")
        glUniform1f(time_location, pygame.time.get_ticks() / 1000.0)
        glUniform2f(resolution_location, display[0], display[1])
        glUniform1i(prev_state_location, 0)
        # glBindTexture(GL_TEXTURE_2D, textures[0])
        glBindTexture(GL_TEXTURE_2D, textures[current_texture])
        # Draw quad
        glBegin(GL_QUADS)
        glVertex2f(-1, -1)
        glVertex2f(1, -1)
        glVertex2f(1, 1)
        glVertex2f(-1, 1)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # 2. Render visualization to screen
    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(visualize_program)

    resolution_location = glGetUniformLocation(visualize_program, "u_resolution")
    glUniform2f(resolution_location, display[0], display[1])

    visualization_state_location = glGetUniformLocation(visualize_program, "u_state")
    glUniform1i(visualization_state_location, 0)
    # glBindTexture(GL_TEXTURE_2D, textures[1])
    glBindTexture(GL_TEXTURE_2D, textures[1-current_texture])
    # Draw quad
    glBegin(GL_QUADS)
    glVertex2f(-1, -1)
    glVertex2f(1, -1)
    glVertex2f(1, 1)
    glVertex2f(-1, 1)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)

    pygame.display.flip()
    pygame.time.wait(50)

    # # Swap textures for next iteration
    current_texture = 1 - current_texture
