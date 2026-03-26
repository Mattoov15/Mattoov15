/*
 * Copyright (c) 2021-present LAAS-CNRS
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 2.1 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU Lesser General Public License for more details.
 *
 *   You should have received a copy of the GNU Lesser General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1
 */

/**
 * @brief  PWM motor control with fixed 50% duty cycle on Ownverter board.
 *         The motor runs autonomously without PC connection.
 *
 * @author Clément Foucher <clement.foucher@laas.fr>
 * @author Luiz Villa <luiz.villa@laas.fr>
 * @author Ayoub Farah Hassan <ayoub.farah-hassan@laas.fr>
 */

/* --------------OWNTECH APIs---------------------------------- */
#include "SpinAPI.h"
#include "TaskAPI.h"

/* --------------SETUP FUNCTIONS DECLARATION------------------- */
void setup_routine();

/* --------------LOOP FUNCTIONS DECLARATION-------------------- */
void loop_critical_task();

/* --------------USER VARIABLES DECLARATIONS------------------- */
static const float32_t duty_cycle = 0.5;

/* --------------SETUP FUNCTIONS------------------------------- */

/**
 * This is the setup routine.
 * We configure PWM at 20 kHz on PWMA with 200 ns dead time
 * between CHA1 and CHA2, and start a critical task at 10 kHz
 * to maintain the duty cycle at 50%.
 */
void setup_routine()
{
    /* Set PWM frequency to 20 kHz */
    spin.pwm.initFixedFrequency(20000);
    /* Timer initialization */
    spin.pwm.initUnit(PWMA);
    /* Set 200 ns dead time between CHA1 and CHA2 */
    spin.pwm.setDeadTime(PWMA, 200, 200);

    /* Start dual output (CHA1 high-side, CHA2 low-side) */
    spin.pwm.startDualOutput(PWMA);

    /* Enable the power driver - activate the optocoupler to take
     * the gate driver out of standby. Without this, no power output
     * on the Ownverter board. Pin 19 = LEG1 driver enable. */
    spin.gpio.configurePin(19, OUTPUT);
    spin.gpio.setPin(19);

    /* Create and start the critical task at 10 kHz (100 µs period) */
    task.createCritical(loop_critical_task, 100);
    task.startCritical();
}

/* --------------LOOP FUNCTIONS-------------------------------- */

/**
 * Critical task running at 10 kHz in real time.
 * Applies the fixed 50% duty cycle to PWMA.
 */
void loop_critical_task()
{
    spin.pwm.setDutyCycle(PWMA, duty_cycle);
}

/**
 * Main function - starts the setup routine.
 */
int main(void)
{
    setup_routine();

    return 0;
}
